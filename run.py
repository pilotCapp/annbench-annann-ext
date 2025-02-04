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

    # Instantiate a dataset class
    # https://hydra.cc/docs/tutorial/working_directory#original-working-directory
    dataset = annbench.instantiate_dataset(
        name=cfg.dataset.name, path=to_absolute_path(cfg.dataset.path)
    )

    ret_all = []
    for param_index in cfg.algo.param_index:

        # Instantiate a search algorithm class
        algo = annbench.instantiate_algorithm(name=cfg.algo.name)

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

        if cfg.algo.name == "annann" or cfg.algo.name == "annann_2":
            print("algo is annann")
            algo.path = str(p)
        elif cfg.algo.name == "annann_base":
            print("algo is annann_base")
            algo.path = str(p)

        if algo.has_train():
            algo.train(vecs=dataset.vecs_base(), path=str(p))
            algo.add(vecs=dataset.vecs_base())
            algo.write(path=str(p))
            

        if not os.path.exists(os.path.join(p, "index.pickle")):
            log.info("The index does not exist")
            algo.add(vecs=dataset.vecs_base())
            algo.write(path=str(p))
        else:
            log.info("The index exists")
            algo.read(path=str(p), D=dataset.D())
        if (
            cfg.algo.name == "annann"
            or cfg.algo.name == "annann_2"
            or cfg.algo.name == "linear_adaptive"
        ):
            algo.normalizer.fit(dataset.vecs_base())

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
                        topk=cfg.topk,
                        r=cfg.r,
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
                    "recall_topk": int(cfg.topk),
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
    if algo.query_time_1 > 0:
        algo.time_log()


if __name__ == "__main__":
    main()
