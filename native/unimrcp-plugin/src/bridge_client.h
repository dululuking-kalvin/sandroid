/*
 * Sandroid MRCP plugin <-> Python bridge client (header).
 * See docs/mrcp-bridge-protocol.md for wire format.
 */

#ifndef SANDROID_BRIDGE_CLIENT_H
#define SANDROID_BRIDGE_CLIENT_H

#include <apr.h>
#include <apr_pools.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Frame types — match docs/mrcp-bridge-protocol.md. */
#define SBR_FRAME_START   0x01
#define SBR_FRAME_AUDIO   0x02
#define SBR_FRAME_EOS     0x03
#define SBR_FRAME_STOP    0x04
#define SBR_FRAME_RESULT  0x10
#define SBR_FRAME_ERROR   0x11

/* Max frame size enforced both sides (1 MiB). */
#define SBR_MAX_FRAME_SIZE (1024 * 1024)

/* Opaque connection handle. */
typedef struct sbr_conn_t sbr_conn_t;

/*
 * Connect to the Python bridge Unix socket.
 *   pool         — APR pool for the connection lifetime
 *   socket_path  — UDS path (NULL → env SANDROID_BRIDGE_SOCK or
 *                  "/var/run/sandroid/bridge.sock")
 *   out_conn     — populated on success
 * Returns 0 on success, -1 on failure (connection not established; caller
 * falls back to canned result.xml behaviour).
 */
int sbr_connect(apr_pool_t *pool, const char *socket_path, sbr_conn_t **out_conn);

/*
 * Send START frame. Non-blocking-friendly; returns 0 on success, -1 on
 * unrecoverable write failure (caller should treat connection as dead).
 */
int sbr_send_start(sbr_conn_t *conn,
                   const char *channel_id,
                   const char *session_id,
                   unsigned int sample_rate,
                   const char *codec);

/*
 * Send one AUDIO frame. Non-blocking: on EAGAIN the frame is dropped and
 * the internal drop counter is incremented. Returns 0 always (drops are
 * not errors); returns -1 only on fatal socket errors.
 */
int sbr_send_audio(sbr_conn_t *conn, const void *pcm, apr_size_t nbytes);

/* Send EOS. Blocking with a short timeout. */
int sbr_send_eos(sbr_conn_t *conn);

/* Send STOP. Best-effort; failure is ignored. */
int sbr_send_stop(sbr_conn_t *conn);

/*
 * Wait for a RESULT frame (blocking, bounded by timeout_ms).
 *   out_nlsml   — on success, pointer to NLSML body in pool memory
 *   out_nbytes  — length of NLSML body
 * Returns  0  success (RESULT received)
 *         -1  timeout, connection closed, or peer sent ERROR
 *         -2  protocol violation (wrong frame type, oversized, etc.)
 */
int sbr_wait_result(sbr_conn_t *conn,
                    apr_pool_t *result_pool,
                    int timeout_ms,
                    const char **out_nlsml,
                    apr_size_t *out_nbytes);

/* Close the socket and release resources. */
void sbr_close(sbr_conn_t *conn);

/* Telemetry: number of AUDIO frames dropped due to EAGAIN. */
apr_uint64_t sbr_dropped_frames(const sbr_conn_t *conn);

#ifdef __cplusplus
}
#endif

#endif /* SANDROID_BRIDGE_CLIENT_H */
