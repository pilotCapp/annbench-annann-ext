import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import yaml
import annbench

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config_plot")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    out = Path(to_absolute_path(cfg.output))
    result_img = Path(to_absolute_path(cfg.result_img))
    result_img.mkdir(exist_ok=True, parents=True)  # Make sure the result_img directory exists
    recall_topk = 1
    
    for p_dataset in sorted(out.glob("*")):
        if p_dataset.is_file() or p_dataset.name.startswith("."):
            continue
        lines = []
        for p_algo in sorted(p_dataset.glob("*")):
            if p_algo.is_file() or p_algo.name.startswith("."):
                continue
            log.info(f"Check {(p_algo / 'result.yaml').resolve()}")
            with (p_algo / "result.yaml").open("rt") as f:
                ret_all = yaml.safe_load(f)
            for ret in ret_all:
                # "ret" is for one param_index. "ret" contains several results for each param_query
                xs, ys, ctrls = [], [], []  # recall, QPS, query_param
                recall_topk = 1
                for r in ret:
                    xs.append(r["recall"])
                    ys.append(1.0 / r["runtime_per_query"])
                    ctrls.append(list(r['param_query'].values())[0])  # Just extract a value
                    recall_topk = r["recall_topk"]
                line = {
                    "xs": xs, "ys": ys, "ctrls": ctrls,
                    "ctrl_label": list(ret[0]['param_query'])[0],  # Just extract the name of query param
                    "label": p_algo.name + "(" + annbench.util.stringify_dict(d=ret[0]['param_index']) + ")"
                }
                lines.append(line)

        log.info(f"Write on {result_img.resolve()}")

        # Save the image on the result_img directory and a working directory (./log) as a log
        for path_img in [result_img / f"{p_dataset.name}.png", f"{p_dataset.name}.png"]:
            annbench.vis.draw(lines=lines, xlabel=r"recall@"+str(recall_topk), ylabel="query/sec (1/s)", title=p_dataset.name,
                              filename=path_img, with_ctrl=cfg.with_query_param, width=cfg.width, height=cfg.height)

        # Save the summary on the result_img directory and a working directory (./log) as a log
        for path_summary in [result_img / f"{p_dataset.name}.yaml", f"{p_dataset.name}.yaml"]:
            with open(path_summary, "wt") as f:
                yaml.dump(lines, f)


if __name__ == "__main__":
    main()