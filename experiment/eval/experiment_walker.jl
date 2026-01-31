"""
```
data_path_collecter(EXPORTED_PREDS_DIR)
```
returns a list with named tuple entries:
```
(;repo, dataset, split_idx, pred_dir, obs_test_file, obs_train_file, PRED_FORMAT)
```
"""
function data_path_collecter(EXPORTED_PREDS_DIR)
    EVAL_DIR = @__DIR__
    EXPERIMENT_DIR = EVAL_DIR |> splitdir |> first
    DATA_DIR = joinpath(EXPERIMENT_DIR, "data")
    PRED_FORMAT = readlines(joinpath(EXPORTED_PREDS_DIR, "format.txt")) |> first

    res_list = []

    _, repos, _ = first(walkdir(EXPORTED_PREDS_DIR))
    for repo in repos
        if startswith(repo, r"[._]") continue end #skip hidden and technical folders

        preds_repo = joinpath(EXPORTED_PREDS_DIR, repo)
        data_repo = joinpath(DATA_DIR, repo, "exported_splits")

        _, datasets, _ = first(walkdir(preds_repo))
        for dataset in datasets 
            pred_dataset = joinpath(preds_repo, dataset)
            data_dataset = joinpath(data_repo, dataset)
            _, splits, _ = first(walkdir(pred_dataset))
            for split in splits
                split_idx = only(match(r"split_(\d+)", split).captures)
                pred_dir = joinpath(pred_dataset, split)
                data_split = joinpath(data_dataset, split)

                obs_test_file = joinpath(data_split, "y_test.csv")
                obs_train_file = joinpath(data_split, "y_train.csv")

                push!(res_list, (;repo, dataset, split_idx, pred_dir, obs_test_file, obs_train_file, PRED_FORMAT))
            end
        end
    end
    res_list
end