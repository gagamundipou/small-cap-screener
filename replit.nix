{pkgs}: {
  deps = [
    pkgs.rustc
    pkgs.libiconv
    pkgs.cargo
    pkgs.postgresql
    pkgs.zlib
    pkgs.xcodebuild
    pkgs.glibcLocales
  ];
}
