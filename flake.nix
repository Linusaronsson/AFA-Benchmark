{
  description = "AFABench development shell";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };
  outputs =
    { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              # direnv otherwise exposes stdenv's Bash without Readline or completion.
              bashInteractive
              git
              just
              uv
              basedpyright
              # Launcher scripts use the host Python; uv installs the exact
              # project interpreter and dependencies from uv.lock.
              python312
            ];
            shellHook = ''
              unset PYTHONPATH
              # Native PyPI wheels need libraries outside their bundled runtime.
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              # GPU drivers belong to the host, not the Python environment.
              if [ -d /run/opengl-driver/lib ]; then
                export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib"
              fi
            '';
          };
        });
    };
}
