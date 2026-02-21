{
  description = "Python dev shell";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };
  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          copier
          pre-commit
          just
          uv
          basedpyright
          ruff
          nextflow
          nf-test
        ];
        shellHook = ''
          unset PYTHONPATH
        '';
      };
    };
}
