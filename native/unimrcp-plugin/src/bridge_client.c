/*
 * Sandroid MRCP plugin <-> Python bridge client.
 *
 * See docs/mrcp-bridge-protocol.md for the wire protocol. This
 * implementation is dependency-free: hand-rolled CBOR encoder for the
 * handful of fields we actually send, raw AF_UNIX sockets, APR for
 * memory + portability niceties only.
 */

#include "bridge_client.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <apr_errno.h>
#include <apr_strings.h>

#define SBR_DEFAULT_SOCKET "/var/run/sandroid/bridge.sock"

/* Short connect timeout — if Python isn't up we want fast fallback. */
#define SBR_CONNECT_TIMEOUT_MS 250
/* Brief write timeout for EOS/STOP/START. */
#define SBR_CONTROL_WRITE_TIMEOUT_MS 500

struct sbr_conn_t {
	int              fd;
	apr_pool_t      *pool;
	apr_uint64_t     dropped;
};

/* ---------- low-level I/O helpers ---------- */

static int set_nonblock(int fd, int on)
{
	int flags = fcntl(fd, F_GETFL, 0);
	if (flags < 0) return -1;
	if (on) flags |= O_NONBLOCK;
	else    flags &= ~O_NONBLOCK;
	return fcntl(fd, F_SETFL, flags);
}

/* Write all bytes or fail. Blocking, bounded by timeout_ms. */
static int write_all_timeout(int fd, const void *buf, size_t nbytes, int timeout_ms)
{
	const char *p = (const char *)buf;
	size_t left = nbytes;
	while (left > 0) {
		ssize_t n = write(fd, p, left);
		if (n > 0) {
			p    += n;
			left -= (size_t)n;
			continue;
		}
		if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
			struct pollfd pfd = { .fd = fd, .events = POLLOUT };
			int pr = poll(&pfd, 1, timeout_ms);
			if (pr <= 0) return -1;  /* timeout or error */
			continue;
		}
		if (n < 0 && errno == EINTR) continue;
		return -1;
	}
	return 0;
}

/* Write as much as possible non-blocking; drop remainder on EAGAIN. */
static int write_nonblock_or_drop(int fd, const void *buf, size_t nbytes)
{
	const char *p = (const char *)buf;
	size_t left = nbytes;
	while (left > 0) {
		ssize_t n = write(fd, p, left);
		if (n > 0) { p += n; left -= (size_t)n; continue; }
		if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return 1;  /* dropped */
		if (n < 0 && errno == EINTR) continue;
		return -1;  /* fatal */
	}
	return 0;
}

/* Read exactly nbytes or fail/timeout. */
static int read_exact_timeout(int fd, void *buf, size_t nbytes, int timeout_ms)
{
	char *p = (char *)buf;
	size_t left = nbytes;
	while (left > 0) {
		struct pollfd pfd = { .fd = fd, .events = POLLIN };
		int pr = poll(&pfd, 1, timeout_ms);
		if (pr <= 0) return -1;
		ssize_t n = read(fd, p, left);
		if (n > 0) { p += n; left -= (size_t)n; continue; }
		if (n == 0) return -1;  /* peer closed */
		if (n < 0 && errno == EINTR) continue;
		if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) continue;
		return -1;
	}
	return 0;
}

/* ---------- frame codec ---------- */

/* Encode length(4B BE) + type(1B). Caller writes payload next. */
static void frame_header(unsigned char out[5], apr_uint32_t payload_len, unsigned char type)
{
	apr_uint32_t total = payload_len + 1;  /* type is part of length */
	out[0] = (unsigned char)((total >> 24) & 0xFF);
	out[1] = (unsigned char)((total >> 16) & 0xFF);
	out[2] = (unsigned char)((total >> 8)  & 0xFF);
	out[3] = (unsigned char)((total      ) & 0xFF);
	out[4] = type;
}

/* ---------- hand-rolled CBOR encoder (minimal) ---------- */

/*
 * We encode a single flavour: map(n) of text-string keys to values that
 * are either text-string or unsigned int. That covers START entirely.
 * Buffer must be pre-sized by the caller.
 */

typedef struct {
	unsigned char *buf;
	size_t         cap;
	size_t         len;
} cbor_w_t;

static int cbor_put(cbor_w_t *w, const void *src, size_t n)
{
	if (w->len + n > w->cap) return -1;
	memcpy(w->buf + w->len, src, n);
	w->len += n;
	return 0;
}

/* CBOR major types: 0=uint, 3=tstr, 5=map. */
static int cbor_write_head(cbor_w_t *w, unsigned char major, apr_uint64_t val)
{
	unsigned char b;
	if (val < 24) {
		b = (unsigned char)((major << 5) | (unsigned char)val);
		return cbor_put(w, &b, 1);
	}
	if (val <= 0xFF) {
		b = (unsigned char)((major << 5) | 24);
		unsigned char v = (unsigned char)val;
		return cbor_put(w, &b, 1) || cbor_put(w, &v, 1);
	}
	if (val <= 0xFFFF) {
		b = (unsigned char)((major << 5) | 25);
		unsigned char v[2] = { (unsigned char)(val >> 8), (unsigned char)val };
		return cbor_put(w, &b, 1) || cbor_put(w, v, 2);
	}
	if (val <= 0xFFFFFFFFULL) {
		b = (unsigned char)((major << 5) | 26);
		unsigned char v[4] = {
			(unsigned char)(val >> 24), (unsigned char)(val >> 16),
			(unsigned char)(val >> 8),  (unsigned char)(val) };
		return cbor_put(w, &b, 1) || cbor_put(w, v, 4);
	}
	return -1;  /* we never send 64-bit values */
}

static int cbor_tstr(cbor_w_t *w, const char *s)
{
	size_t n = strlen(s);
	if (cbor_write_head(w, 3, n) < 0) return -1;
	return cbor_put(w, s, n);
}

static int cbor_uint(cbor_w_t *w, apr_uint64_t v)
{
	return cbor_write_head(w, 0, v);
}

static int cbor_map_header(cbor_w_t *w, apr_uint64_t n)
{
	return cbor_write_head(w, 5, n);
}

/* ---------- connect ---------- */

int sbr_connect(apr_pool_t *pool, const char *socket_path, sbr_conn_t **out_conn)
{
	const char *path = socket_path;
	if (!path) path = getenv("SANDROID_BRIDGE_SOCK");
	if (!path) path = SBR_DEFAULT_SOCKET;

	int fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd < 0) return -1;

	struct sockaddr_un addr;
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	if (strlen(path) >= sizeof(addr.sun_path)) {
		close(fd);
		return -1;
	}
	strcpy(addr.sun_path, path);

	/* Non-blocking connect so we can bound the connect wait. */
	if (set_nonblock(fd, 1) < 0) { close(fd); return -1; }

	int rc = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
	if (rc < 0 && errno != EINPROGRESS) { close(fd); return -1; }
	if (rc < 0) {
		struct pollfd pfd = { .fd = fd, .events = POLLOUT };
		int pr = poll(&pfd, 1, SBR_CONNECT_TIMEOUT_MS);
		if (pr <= 0) { close(fd); return -1; }
		int err = 0;
		socklen_t el = sizeof(err);
		if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &el) < 0 || err != 0) {
			close(fd); return -1;
		}
	}

	/* Stay non-blocking — AUDIO path needs EAGAIN drop behaviour. */
	sbr_conn_t *c = apr_pcalloc(pool, sizeof(*c));
	c->fd = fd;
	c->pool = pool;
	c->dropped = 0;
	*out_conn = c;
	return 0;
}

/* ---------- senders ---------- */

int sbr_send_start(sbr_conn_t *conn,
                   const char *channel_id,
                   const char *session_id,
                   unsigned int sample_rate,
                   const char *codec)
{
	if (!conn || conn->fd < 0) return -1;
	unsigned char payload[512];
	cbor_w_t w = { .buf = payload, .cap = sizeof(payload), .len = 0 };
	if (cbor_map_header(&w, 4) < 0) return -1;
	if (cbor_tstr(&w, "channel_id")  < 0 || cbor_tstr(&w, channel_id ? channel_id : "") < 0) return -1;
	if (cbor_tstr(&w, "session_id")  < 0 || cbor_tstr(&w, session_id ? session_id : "") < 0) return -1;
	if (cbor_tstr(&w, "sample_rate") < 0 || cbor_uint(&w, sample_rate) < 0) return -1;
	if (cbor_tstr(&w, "codec")       < 0 || cbor_tstr(&w, codec ? codec : "LPCM") < 0) return -1;

	unsigned char header[5];
	frame_header(header, (apr_uint32_t)w.len, SBR_FRAME_START);

	/* START is small and important — write blocking with short timeout. */
	if (write_all_timeout(conn->fd, header, 5, SBR_CONTROL_WRITE_TIMEOUT_MS) < 0) return -1;
	if (write_all_timeout(conn->fd, payload, w.len, SBR_CONTROL_WRITE_TIMEOUT_MS) < 0) return -1;
	return 0;
}

int sbr_send_audio(sbr_conn_t *conn, const void *pcm, apr_size_t nbytes)
{
	if (!conn || conn->fd < 0) return -1;
	if (nbytes == 0) return 0;
	if (nbytes > SBR_MAX_FRAME_SIZE - 1) return -1;

	unsigned char header[5];
	frame_header(header, (apr_uint32_t)nbytes, SBR_FRAME_AUDIO);

	int rc = write_nonblock_or_drop(conn->fd, header, 5);
	if (rc < 0) return -1;
	if (rc == 1) { conn->dropped++; return 0; }

	rc = write_nonblock_or_drop(conn->fd, pcm, nbytes);
	if (rc < 0) return -1;
	if (rc == 1) conn->dropped++;
	return 0;
}

int sbr_send_eos(sbr_conn_t *conn)
{
	if (!conn || conn->fd < 0) return -1;
	unsigned char header[5];
	frame_header(header, 0, SBR_FRAME_EOS);
	return write_all_timeout(conn->fd, header, 5, SBR_CONTROL_WRITE_TIMEOUT_MS);
}

int sbr_send_stop(sbr_conn_t *conn)
{
	if (!conn || conn->fd < 0) return 0;  /* best-effort */
	unsigned char header[5];
	frame_header(header, 0, SBR_FRAME_STOP);
	(void)write_all_timeout(conn->fd, header, 5, SBR_CONTROL_WRITE_TIMEOUT_MS);
	return 0;
}

/* ---------- receive RESULT ---------- */

int sbr_wait_result(sbr_conn_t *conn,
                    apr_pool_t *result_pool,
                    int timeout_ms,
                    const char **out_nlsml,
                    apr_size_t *out_nbytes)
{
	if (!conn || conn->fd < 0) return -1;

	/* Loop: skip unknown types; surface RESULT / ERROR. */
	struct timeval start, now;
	gettimeofday(&start, NULL);

	for (;;) {
		unsigned char header[5];
		/* How much budget is left? */
		gettimeofday(&now, NULL);
		long elapsed_ms = (now.tv_sec - start.tv_sec) * 1000L +
		                  (now.tv_usec - start.tv_usec) / 1000L;
		int remain_ms = timeout_ms - (int)elapsed_ms;
		if (remain_ms <= 0) return -1;

		if (read_exact_timeout(conn->fd, header, 5, remain_ms) < 0) return -1;
		apr_uint32_t total =
			((apr_uint32_t)header[0] << 24) |
			((apr_uint32_t)header[1] << 16) |
			((apr_uint32_t)header[2] << 8)  |
			((apr_uint32_t)header[3]);
		unsigned char type = header[4];
		if (total < 1 || total - 1 > SBR_MAX_FRAME_SIZE) return -2;
		apr_uint32_t payload_len = total - 1;

		char *payload = NULL;
		if (payload_len > 0) {
			payload = apr_palloc(result_pool, payload_len);
			gettimeofday(&now, NULL);
			elapsed_ms = (now.tv_sec - start.tv_sec) * 1000L +
			             (now.tv_usec - start.tv_usec) / 1000L;
			remain_ms = timeout_ms - (int)elapsed_ms;
			if (remain_ms <= 0) return -1;
			if (read_exact_timeout(conn->fd, payload, payload_len, remain_ms) < 0) return -1;
		}

		if (type == SBR_FRAME_RESULT) {
			*out_nlsml  = payload;
			*out_nbytes = payload_len;
			return 0;
		}
		if (type == SBR_FRAME_ERROR) {
			return -1;
		}
		/* Unknown frame type — ignore per forward-compat rule and loop. */
	}
}

/* ---------- lifecycle ---------- */

void sbr_close(sbr_conn_t *conn)
{
	if (!conn) return;
	if (conn->fd >= 0) {
		close(conn->fd);
		conn->fd = -1;
	}
}

apr_uint64_t sbr_dropped_frames(const sbr_conn_t *conn)
{
	return conn ? conn->dropped : 0;
}
