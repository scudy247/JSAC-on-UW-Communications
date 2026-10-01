#!/usr/bin/env bash
# Build one of our DESERT add-ons (desert/addon/<name>) into OUR DESERT install (THEORY-B T-9), the
# way DESERT's installer builds add-ons (DESERT_Framework/Installer/installDESERT_LOCAL.sh,
# build_DESERT_addon, no-WOSS branch): copy the sources to <prefix>/DESERT-4.0.0-ADDONS-src/<name>,
# autogen there, configure out of source in <prefix>/DESERT-4.0.0-ADDONS-build/<name> against the install's ns-allinone / nsmiracle / DESERT trees, make, make install
# (puts lib<name>.so* into <prefix>/lib, i.e. the developers' "copy the .so to lib/" step, with symlinks).
# Never targets the shared tree ~/Underwater/DESERT_Underwater (MERMAID's uncommitted edits live there).
#
#   desert/build_addon.sh uwjsac_hello        # log: <prefix>/DESERT-4.0.0-ADDONS-src/<name>.build.log
set -euo pipefail
name="${1:?usage: build_addon.sh <addon-name>}"
here="$(cd "$(dirname "$0")" && pwd)"
prefix="${UWSB_DESERT_PREFIX:-$HOME/Underwater/DESERT_JSAC/DESERT_buildCopy_LOCAL}"
prefix="$(cd "$prefix" && pwd -P)"
case "$prefix" in
  "$(cd "$HOME/Underwater" && pwd -P)/DESERT_Underwater"*) echo "refusing: $prefix is the shared DESERT tree" >&2; exit 1 ;;
esac
src="$here/addon/$name"
[ -d "$src" ] || { echo "no add-on $src" >&2; exit 1; }
bh="$prefix/.buildHost"
dst="$prefix/DESERT-4.0.0-ADDONS-src/$name"
bld="$prefix/DESERT-4.0.0-ADDONS-build/$name"
log="$prefix/DESERT-4.0.0-ADDONS-src/$name.build.log"
mkdir -p "$prefix/DESERT-4.0.0-ADDONS-src" "$prefix/DESERT-4.0.0-ADDONS-build"
rm -rf "$dst" "$bld" && cp -r "$src" "$dst" && mkdir "$bld"
{
  echo "== $(date -Is) build $name from $src into $prefix"
  (cd "$dst" && ./autogen.sh)
  cd "$bld"
  CXXFLAGS="-Wno-write-strings" CFLAGS="-Wno-write-strings" "$dst/configure" \
      --with-ns-allinone="$bh" \
      --with-nsmiracle="$bh/nsmiracle-2.0.0" \
      --with-desert="$prefix/DESERT-4.0.0-src" \
      --with-desert-build="$prefix/DESERT-4.0.0-build" \
      --with-desertAddon="$prefix/DESERT-4.0.0-ADDONS-src" \
      --with-desertAddon-build="$prefix/DESERT-4.0.0-ADDONS-build" \
      --prefix="$prefix"
  make -j 2
  [ -s initTcl.cc ] && ! grep -q 'code\[\] = "";' initTcl.cc || { echo "empty Tcl init code"; exit 1; }
  make install
  echo "== $(date -Is) done"
} > "$log" 2>&1 || { echo "build failed, see $log" >&2; tail -n 30 "$log" >&2; exit 1; }
echo "built and installed $name (log $log)"
ls -l "$prefix/lib/lib${name//_/}".so*
