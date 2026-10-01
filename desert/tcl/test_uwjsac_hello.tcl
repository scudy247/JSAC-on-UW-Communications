# Load test for the uwjsac_hello add-on: the library loads, the class instantiates, its Tcl
# commands answer. Run with our DESERT's ns (environment of DESERT_JSAC):  ns test_uwjsac_hello.tcl
load libMiracle.so
load libuwjsachello.so

set ns [new Simulator]
# class default from uwjsac-hello-init.tcl (embedded in the library): must exist before any instance
if {[catch {Module/UW/JSAC/HELLO set debug_} d] || $d != 0} { puts "FAIL: no class default debug_"; exit 1 }
set m [new Module/UW/JSAC/HELLO]
puts "hello: [$m hello]"
puts "debug_: [$m set debug_]"
puts "counts: up [$m getUpCount] down [$m getDownCount]"
puts "UWJSAC_HELLO_OK"
