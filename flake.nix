{
  description = "Introduction to Machine Learning";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs   = import nixpkgs { inherit system; };
    python = pkgs.python3;
    pyPkgs = python.pkgs;

    versionPioneer = pyPkgs.buildPythonPackage rec {
      pname    = "version-pioneer";
      version  = "v0.0.13";

      src = pkgs.fetchFromGitHub {
        owner  = "kiyoon";
        repo   = "version-pioneer";
        rev    = "v0.0.13";
        sha256 = "sha256-DYX+VpKXJrKIHcJRMk4ZYgrQXr0181Jsv5eny7yLVpo=";
      };

      nativeBuildInputs = with pyPkgs; [
        hatchling
        hatch-requirements-txt
        tomli # for Python < 3.11
      ];

      buildInputs = [];
      propagatedBuildInputs = [];
      checkInputs = [ pyPkgs.pytest ];

      format = "pyproject";
    };

    jupynium = pyPkgs.buildPythonPackage rec {
      pname    = "jupynium";
      version  = "v0.2.6";

      src = pkgs.fetchFromGitHub {
        owner  = "kiyoon";
        repo   = "jupynium.nvim";
        rev    = "v0.2.6";
        sha256 = "sha256-+9J9v+r3fqPWZuQotgzKgqu0/jmviIDeweUUIb4Lxmc=";
      };

      nativeBuildInputs = with pyPkgs; [
        hatchling
        hatch-requirements-txt
        versionPioneer
      ];

      propagatedBuildInputs = with pyPkgs; [
        selenium
        coloredlogs
        gitpython
        persist-queue
        pynvim
        verboselogs
        nbclassic
      ];

      checkInputs = [ pyPkgs.pytest ];
      format = "pyproject";
    };

    # https://github.com/NixOS/nixpkgs/pull/268078
    jupyterEnv = pkgs.python3.withPackages (ps: with ps; [
      notebook nbclassic
    ]);
  in {
    packages.${system} = {
      jupynium        = jupynium;
      version-pioneer = versionPioneer;
    };

    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [
        jupynium
        jupyterEnv
        pkgs.firefox
        pkgs.pyright

        pyPkgs.numpy
        pyPkgs.pandas
        pyPkgs.seaborn
        pyPkgs.scikit-learn

        pyPkgs.matplotlib
      ];
    };
  };
}
