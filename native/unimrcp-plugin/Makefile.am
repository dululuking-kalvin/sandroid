AM_CPPFLAGS                = $(UNIMRCP_PLUGIN_INCLUDES)

plugin_LTLIBRARIES         = sandroidrecog.la

sandroidrecog_la_SOURCES   = src/sandroid_recog_engine.c \
                             src/bridge_client.c \
                             src/bridge_client.h
sandroidrecog_la_LDFLAGS   = $(UNIMRCP_PLUGIN_OPTS)

include $(top_srcdir)/build/rules/uniplugin.am
