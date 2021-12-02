kazoeage-oneesan-cuda
======

GPGPU version of [kazoeage-oneesan](https://github.com/primenumber/kazoeage-oneesan)

## Build

Required: NVIDIA GPU (Pascal or later), CUDA Toolkit

```
$ make
```

## Usage

`./oneesan N expand`
N: grid size
expand: CPU collect all `expand` step path from start, GPU will starts from each path

```
$ ./oneesan 6 20
oneesan(6) = 575780564
```

## License

MIT License
