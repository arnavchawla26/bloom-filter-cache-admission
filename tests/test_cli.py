from cache_admission.cli import build_parser, main


def test_benchmark_subcommand_runs_and_prints_table(capsys):
    code = main(["benchmark", "--workload", "zipf", "--requests", "500", "--items", "50",
                 "--capacity", "10", "--seed", "1"])
    assert code == 0
    out = capsys.readouterr().out
    assert "TinyLFU" in out
    assert "hit_rate" in out


def test_benchmark_subcommand_shifting_workload(capsys):
    code = main(["benchmark", "--workload", "shifting", "--requests", "800", "--items", "40",
                 "--capacity", "8", "--phases", "4", "--seed", "2"])
    assert code == 0
    assert "workload=shifting" in capsys.readouterr().out


def test_benchmark_subcommand_scan_workload(capsys):
    code = main(["benchmark", "--workload", "scan", "--requests", "400", "--items", "15",
                 "--scan-length", "100", "--capacity", "15", "--seed", "3"])
    assert code == 0
    assert "workload=scan" in capsys.readouterr().out


def test_sketch_demo_subcommand_runs_and_prints_table(capsys):
    code = main(["sketch-demo", "--requests", "2000", "--items", "100", "--seed", "1"])
    assert code == 0
    out = capsys.readouterr().out
    assert "estimate" in out
    assert "mode=conservative" in out


def test_sketch_demo_naive_flag(capsys):
    code = main(["sketch-demo", "--requests", "500", "--items", "50", "--naive", "--seed", "1"])
    assert code == 0
    assert "mode=naive" in capsys.readouterr().out


def test_build_parser_requires_a_subcommand():
    parser = build_parser()
    assert parser.parse_args(["benchmark"]).command == "benchmark"
    assert parser.parse_args(["sketch-demo"]).command == "sketch-demo"
