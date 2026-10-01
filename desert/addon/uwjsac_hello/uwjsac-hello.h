// uwjsac_hello: empty pass-through ns-miracle module (Proposal B, DESERT build test, Phase 1-D task 1).
// Proves that a new add-on of ours compiles against DESERT 4.0.0, installs into our own DESERT
// install (DESERT_JSAC, THEORY-B T-9) and loads in ns. Packets are forwarded unchanged in their
// direction of travel; the module only counts them. Template: DESERT_Addons/packer_uwcbr.
#ifndef UWJSAC_HELLO_H
#define UWJSAC_HELLO_H

#include <module.h>

class UwJsacHello : public Module
{
public:
	UwJsacHello();
	virtual ~UwJsacHello() = default;
	virtual int command(int argc, const char *const *argv);
	virtual void recv(Packet *p);

protected:
	int debug_;
	int n_up_;
	int n_down_;
};

#endif
