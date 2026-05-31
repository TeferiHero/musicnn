{ nixpkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

nixpkgs.mkShell {
  nativeBuildInputs = with nixpkgs; [
     uv
  ];
  shellHook = ''
    export CUDA_PATH=${nixpkgs.cudatoolkit}
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:${nixpkgs.cudaPackages.cudnn.lib}:${nixpkgs.cudatoolkit.lib}:${nixpkgs.stdenv.cc.cc.lib}:$LD_LIBRARY_PATH
  '';
}
