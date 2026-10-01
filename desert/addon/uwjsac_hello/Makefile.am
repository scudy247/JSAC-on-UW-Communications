# uwjsac_hello add-on (structure copied from DESERT_Addons/packer_uwcbr/Makefile.am).
AM_CXXFLAGS = -Wall -ggdb3

lib_LTLIBRARIES = libuwjsachello.la

libuwjsachello_la_SOURCES = uwjsac-hello.cpp uwjsac-hello.h initlib.cc

libuwjsachello_la_CPPFLAGS = @NS_CPPFLAGS@ @NSMIRACLE_CPPFLAGS@ @DESERT_CPPFLAGS@
libuwjsachello_la_LDFLAGS =  @NS_LDFLAGS@ @NSMIRACLE_LDFLAGS@ @DESERT_LDFLAGS@ @DESERT_LDFLAGS_BUILD@
libuwjsachello_la_LIBADD = @NS_LIBADD@ @NSMIRACLE_LIBADD@ @DESERT_LIBADD@

nodist_libuwjsachello_la_SOURCES = initTcl.cc

BUILT_SOURCES = initTcl.cc

CLEANFILES = initTcl.cc

TCL_FILES = uwjsac-hello-init.tcl

initTcl.cc: Makefile $(TCL_FILES)
# $(srcdir), not the template's $(VPATH): VPATH is empty in an in-source build, which silently
# embedded an empty Tcl init (found in the first build test).
		cat $(srcdir)/$(TCL_FILES) | @TCL2CPP@ UwjsacHelloTclCode > initTcl.cc

EXTRA_DIST = $(TCL_FILES)
