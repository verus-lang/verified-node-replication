# Verified NR Benchmarks

The following benchmarks are used in the evaluation for the paper "Verus: A Practical Foundation for Systems Verification" (SOSP 2024).

## Dependencies

To run the benchmarks, you need to install the following dependencies:

```
sudo apt-get install curl wget liburcu-dev libhwloc-dev python3-venv  texlive-xetex texlive-fonts-extra pkg-config clang make g++
```

Install Rust using rustup:

```shell
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
```


Linear Dafny requires a specific version of libssl. You can install this with the follwing command:
```
wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb
sudo dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb
rm -rf libssl1.1_1.1.0g-2ubuntu4_amd64.deb
```

## Running

To run the benchmarks run the following command:

```shell
bash run_benchmarks.sh
```

This will run three configurations:
1) Verus NR
2) IronSync NR
3) Rust NR (unverified)

Note: for IronSync NR and Rust NR, we use the same codebase as IronSync.

## Results

The results will be shown as a graph. See `nr-results-throughput-vs-cores-numa-fill.pdf`
or `nr-results-throughput-vs-cores-numa-fill.png`.
