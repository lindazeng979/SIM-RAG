# SIM-RAG: Self-practicing for Inner Monologue-based Retrieval Augmented Generation

This is the repository for the paper: Knowing You Don’t Know: Learning When to Continue Search in
Multi-round RAG through Self-Practicing (SIGIR '25). It provides a framework to run SIM-RAG experiments and evaluate models. Follow the steps below to set up and run your experiments.

## Prerequisites

1. Download model checkpoints, the retrieval corpus, and datasets from here: TBD

2. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your/repository.git
   cd repository
   ```

3. Ensure you have all the required dependencies installed (refer to `requirements.txt` or installation instructions in the repo).

4. If you're using GPT, make sure to set your API key in the environment. You can do this by adding the following line to your .bashrc, .zshrc, or equivalent shell configuration file:

    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```

    Then, run:
     ```bash
    source ~/.bashrc  # or `source ~/.zshrc` for Zsh users
    ```

5. Likewise, if you're using Llama, make sure to set the local path to Llama in the environment:
    
    ```bash
    export LLAMA_PATH="/path/to/your/llama"
    ```

    Then, run:
     ```bash
    source ~/.bashrc  # or `source ~/.zshrc` for Zsh users
    ```

## Prepare Data
1. Place downloaded retrieval files `corpus.pkl`, `wiki_corpus.pkl`, `retriever_settings.pkl`, and `wiki_retriever_settings.pkl` into the `bm25_search/` directory.

2. Prepare the Original Datasets
Place the downloaded three datasets in the `data/original/` folder. Alternatively, you can download the 2wikimultihopqa dataset from the dataset GitHub and place the folder containing the JSON files in the `data/` directory. Then, run dataset preparation scripts.
```bash
python /data/prepare_2wikimultihopqa.py
python /data/prepare_triviaqa.py
python /data/prepare_hotpotqa.py
```

## Run Experiment

To run the SIM-RAG experiment, you'll first need to create a custom script using `run_SIM-RAG.py`. This script will guide you through entering parameters and generating an executable `.sh` file for your experiment.

1. Run `create_SIM-RAG.py`:
   ```bash
   python3 create_SIM-RAG.py
   ```

2. Follow the prompts to enter details for the SIM-RAG experiment.

3. After you have entered all details, the script will generate a `.sh` file in the `bash_scripts` directory, which can be used to run the SIM-RAG experiment.

4. Change the permissions of the generated `.sh` file to make it executable:
   ```bash
   chmod +x bash_scripts/{script_filename}
   ```

5. Run the generated `.sh` file to start the experiment:
   ```bash
   ./bash_scripts/{script_filename}
   ```

## Evaluating Predictions

Once the SIM-RAG experiment is complete, you can evaluate the predictions using `evaluate_SIM-RAG.py`.

1. The predictions for each dataset are saved in the `predictions/` directory in the format `{name}_predictions.csv`.

2. To evaluate the predictions, run `evaluate_SIM-RAG.py` with the experiment ame:
   ```bash
   python3 evaluate_SIM-RAG.py --experiment_name {name}
   ```

3. The script will output the EM and F1 score of the predictions.

Fine-grained, intermediate evaluation data and statistics can also be found in `logs/{name}_log.txt`

## Customizing Experiments

To run offline, modify the `.sh` file to run each command with nohup. If you have downloaded checkpoints for an already trained Critic, modify the `.sh` to only run the last line (inference) while passing the path to the Critic into `--dm_path`. Make sure the tokenizer is in the same directory.

## Citation 
If you plan to use the code, please cite our paper: TBD.
