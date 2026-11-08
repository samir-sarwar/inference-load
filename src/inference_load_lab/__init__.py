def main() -> None:
    import sys

    if sys.argv[1:2] == ["bench"]:
        from inference_load_lab.benchmark import main as benchmark_main

        benchmark_main(sys.argv[2:])
        return

    from inference_load_lab.client import main as client_main

    client_main()
