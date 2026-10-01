#include "uwjsac-hello.h"

#include <packet.h>
#include <tclcl.h>

static class UwJsacHelloClass : public TclClass
{
public:
	UwJsacHelloClass() : TclClass("Module/UW/JSAC/HELLO") {}
	TclObject *create(int, const char *const *) { return new UwJsacHello(); }
} class_uwjsac_hello;

UwJsacHello::UwJsacHello() : Module(), debug_(0), n_up_(0), n_down_(0)
{
	bind("debug_", &debug_);
}

int
UwJsacHello::command(int argc, const char *const *argv)
{
	Tcl &tcl = Tcl::instance();
	if (argc == 2) {
		if (strcasecmp(argv[1], "hello") == 0) {
			tcl.resultf("uwjsac_hello loaded");
			return TCL_OK;
		}
		if (strcasecmp(argv[1], "getUpCount") == 0) {
			tcl.resultf("%d", n_up_);
			return TCL_OK;
		}
		if (strcasecmp(argv[1], "getDownCount") == 0) {
			tcl.resultf("%d", n_down_);
			return TCL_OK;
		}
	}
	return Module::command(argc, argv);
}

void
UwJsacHello::recv(Packet *p)
{
	if (HDR_CMN(p)->direction() == hdr_cmn::UP) {
		n_up_++;
		sendUp(p);
	} else {
		n_down_++;
		sendDown(p);
	}
}
