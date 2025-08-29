Of course. Here is the updated `README.md` that includes the project's current status and goals.

-----

# Flatland Project

This project involves solving pathfinding challenges within the Flatland 2D grid environment.

-----

## 📊 Current Status

The current performance of the implemented solutions is as follows:

  * **Question 1:** 100% (Perfect Score)
  * **Question 2:** 100% (Perfect Score)
  * **Question 3:** 69%

The detailed performance metrics can be found in `log.txt`.

-----

## 🎯 Goal

The primary objective is to **improve the performance of the solution for Question 3**. This will be achieved by analyzing the existing code, optimizing the environment, and referencing relevant research papers and algorithms for pathfinding and multi-agent systems.

-----

## ⚙️ Setup

To run this project, you need to replicate the specific Conda environment.

1.  **Create the Conda Environment**
    Use the provided `environment.yml` file to create the necessary environment named `flatland-r1`.

    ```bash
    conda env create -f environment.yml
    ```

2.  **Activate the Environment**
    Before running any scripts, activate the newly created environment.

    ```bash
    conda activate flatland-r1
    ```

-----

## ▶️ Usage

The main script is executed as follows. Make sure the `flatland-r1` environment is active.

```bash
python question3.py
```

-----

## 📁 Project Structure

The map data required to run the tests are located in the following directories:

  * `multi_test_case/`: Contains the map data for Question 3.
  * `single_test_case/`: Contains the map data for Questions 1 and 2.
