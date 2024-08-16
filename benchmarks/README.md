# Verified NR Benchmarks

## Dependencies

```
sudo apt install liburcu-dev libhwloc-dev python3-venv  texlive-xetex texlive-fonts-extra
```

Specific libssl version for building Linear Dafny:
```
wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb
sudo dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb
```

## Running

To run the benchmarks run the following command:

```shell
bash run_benchmarks.sh
```

## Results

The results will be stored as a graph. See `nr-results-throughput-vs-cores-numa-fill.pdf`
or `nr-results-throughput-vs-cores-numa-fill.png`
