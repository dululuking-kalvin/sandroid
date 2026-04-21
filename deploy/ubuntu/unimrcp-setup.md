# UniMRCP 1.8.0 on Ubuntu 22.04

Setup notes for Phase 5e Step 1a — real UniMRCP server in a remote Ubuntu
VM, drives our own recognizer plugin via MRCPv2.

## Target

- Ubuntu 22.04 LTS (jammy), kernel 5.15+
- APR 1.7.0, APR-util 1.6.1, Sofia-SIP 1.12.11 (all from apt)
- UniMRCP 1.8.0 installed under `/opt/unimrcp/`

## Install dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential autoconf libtool pkg-config \
    libapr1-dev libaprutil1-dev libsofia-sip-ua-dev \
    tcpdump net-tools git
```

`libtool` the command is absent on 22.04 (only `libtoolize` ships), but
UniMRCP's bootstrap only needs `libtoolize`.

## Sofia-SIP shim

UniMRCP's `configure` accepts `--with-sofia-sip=PATH` and looks for
`$PATH/lib/pkgconfig/sofia-sip-ua.pc`. Ubuntu's `libsofia-sip-ua-dev`
package drops the `.pc` at `/usr/lib/pkgconfig/`, so `--with-sofia-sip=/usr`
is all we need. No `sofia-config` shim script required.

## Clone + patch

```bash
sudo mkdir -p /opt/unimrcp-src
sudo git clone --depth 1 --branch unimrcp-1.8.0 \
    https://github.com/unispeech/unimrcp.git /opt/unimrcp-src
cd /opt/unimrcp-src
sudo patch -p1 < /path/to/sandroid/deploy/unimrcp/patches/0001-drop-apr-pool-mutex-set.patch
```

The patch removes a call to `apr_pool_mutex_set()`, a symbol that only
exists on APR development branches between 1.x and 2.0 and was never
released. Without this, the link step fails on every apt-packaged APR.

## Configure + build + install

```bash
cd /opt/unimrcp-src
sudo ./bootstrap
sudo ./configure --prefix=/opt/unimrcp --with-sofia-sip=/usr
sudo make -j"$(nproc)"
sudo make install
```

Expected configure report (tail):

```
UniMRCP version............... : 1.8.0
APR version................... : 1.7.0
APR-util version.............. : 1.6.1
Sofia-SIP version............. : 1.12.11devel
Demo recognizer plugin........ : yes
Installation directory........ : /opt/unimrcp
```

## Smoke test

```bash
LD_LIBRARY_PATH=/opt/unimrcp/lib /opt/unimrcp/bin/unimrcpserver -v
LD_LIBRARY_PATH=/opt/unimrcp/lib /opt/unimrcp/bin/umc -v
```

Both must print `1.8.0`.

## Layout

```
/opt/unimrcp/
├── bin/        unimrcpserver, umc, unimrcpclient, asrclient
├── conf/       unimrcpserver.xml, unimrcpclient.xml, umc-scenarios/
├── data/       (prompts, grammars — umc test inputs)
├── include/    public headers (we'll include these from our plugin)
├── lib/        libunimrcpserver.so, libmrcp.so, ... (+ pkgconfig)
├── plugin/     demorecog.so, demosynth.so, demoverifier.so, mrcprecorder.so
├── log/        (server runtime logs)
└── var/        (server runtime data)
```

The key artifacts for Step 1a:
- `plugin/demorecog.so` — the shape our own `sandroid-engine` plugin will
  mirror; we will copy and adapt `/opt/unimrcp-src/plugins/demo-recog`.
- `bin/umc` + `conf/umc-scenarios/` — the validation client we use before
  bringing FreeSWITCH into the loop.

## Upgrade path

UniMRCP 1.8.0 is from 2022. If future work wants a newer release (there
is no official successor as of Apr 2026, but the `master` branch has had
fixes), the patch series lives under `deploy/unimrcp/patches/` and can be
replayed against a fresh clone.
