def main() -> None:
    import sys

    if sys.argv[1:2] == ["experiment"]:
        from inference_load_lab.experiment import main as experiment_main

        experiment_main(sys.argv[2:])
        return

    if sys.argv[1:2] == ["analyze"]:
        from inference_load_lab.analysis import main as analysis_main

        analysis_main(sys.argv[2:])
        return

    if sys.argv[1:2] == ["serve"]:
        from inference_load_lab.gateway import main as gateway_main

        gateway_main(sys.argv[2:])
        return

    if sys.argv[1:2] == ["bench"]:
        from inference_load_lab.benchmark import main as benchmark_main

        benchmark_main(sys.argv[2:])
        return

    from inference_load_lab.client import main as client_main

    client_main()
