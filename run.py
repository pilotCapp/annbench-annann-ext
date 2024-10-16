import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import annbench
import yaml
import numpy as np
import time
import os

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config_run")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Instantiate a search algorithm class
    algo = annbench.instantiate_algorithm(name=cfg.algo.name)

    # Instantiate a dataset class
    # https://hydra.cc/docs/tutorial/working_directory#original-working-directory
    dataset = annbench.instantiate_dataset(
        name=cfg.dataset.name, path=to_absolute_path(cfg.dataset.path)
    )

    ret_all = []
    for param_index in cfg.algo.param_index:

        algo.set_index_param(param=param_index)

        log.info(f"Start to build. index_param={param_index}")

        # The absolute path to the index
        p = (
            Path(to_absolute_path(cfg.interim))
            / cfg.dataset.name
            / cfg.algo.name
            / algo.stringify_index_param(param=param_index)
        )
        p.parent.mkdir(
            exist_ok=True, parents=True
        )  # Make sure the parent directory exists

        if algo.name:
            algo.path = str(p)

        algo.normalizer.fit(dataset.vecs_train())

        ## should probably split this for index and model
        # Build the index, or read the index if it has been already built
        if os.path.exists(os.path.join(p, "index.pkl")) and not algo.has_train():
            log.info("The index already exists. Read it")
            algo.read(path=str(p), D=dataset.D())
        else:
            print("index does not exist")
            t0 = time.time()
            memory_usage_before = algo.get_memory_usage()

            log.info("Start to train")
            algo.train(  # training model, checks nativly if model exists
                vecs=dataset.vecs_train(), path=p
            )  # added path to save the model for annan, probably results in error for other algorithms
            log.info("Start to add")
            algo.add(vecs=dataset.vecs_base())
            build_time = time.time() - t0
            index_size = algo.get_memory_usage() - memory_usage_before
            print(f"Index build time: {build_time:.2f} sec")
            print(f"Index size: {index_size / 1024 / 1024:.2f} MB")

            algo.write(path=str(p))

        ret = []
        # Run search for each param_query
        pname, vals = list(cfg.algo.param_query.items())[
            0
        ]  # e.g., pname="search_k", vals=[100, 200, 400]
        for val in vals:
            param_query = {pname: val}  # e.g., param_query={"search_k": 100}
            log.info(f"Start to search. param_query={param_query}")

            # Run cfg.num_trial times, and take the average
            runtime_per_query, recall = np.mean(
                [
                    annbench.util.evaluate(
                        algo=algo,
                        vecs_query=dataset.vecs_query(),
                        gt=dataset.groundtruth(),
                        topk=1,
                        r=1,
                        param_query=param_query,
                    )
                    for _ in range(cfg.num_trial)
                ],
                axis=0,
            )
            ret.append(
                {
                    "param_index": dict(param_index),
                    "param_query": dict(param_query),
                    "runtime_per_query": float(runtime_per_query),
                    "recall": float(recall),
                }
            )
            log.info("Finish")

        ret_all.append(ret)

    # (1) Save the result on the local log directory
    with open("result.yaml", "wt") as f:
        yaml.dump(ret_all, f)

    # (2) And the output directory.
    out = (
        Path(to_absolute_path(cfg.output))
        / cfg.dataset.name
        / cfg.algo.name
        / "result.yaml"
    )
    out.parent.mkdir(
        exist_ok=True, parents=True
    )  # Make sure the parent directory exists
    with out.open("wt") as f:
        yaml.dump(ret_all, f)

    # (3) Print test results
    time_sum = algo.query_time_1 + algo.query_time_2 + algo.query_time_3
    print(f"query time 1%: {algo.query_time_1/time_sum}")
    print(f"query time 2%: {algo.query_time_2/time_sum}")
    print(f"query time 3%: {algo.query_time_3/time_sum}")
    print(f"\nquery time 3_1%: {algo.query_time_3_1/algo.query_time_3}")
    print(f"\nquery time 3_2%: {algo.query_time_3_2/algo.query_time_3}")
    print(f"\nquery time 3_3%: {algo.query_time_3_3/algo.query_time_3}")


if __name__ == "__main__":
    main()
