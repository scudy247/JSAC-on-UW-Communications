#include <tclcl.h>

extern EmbeddedTcl UwjsacHelloTclCode;

extern "C" int
Uwjsachello_Init()
{
	UwjsacHelloTclCode.load();
	return 0;
}
