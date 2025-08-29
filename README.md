Of course, here is the updated `README.md` reflecting the new details about the development process for Question 3.

-----

# Flatland Project

This project involves solving pathfinding challenges within the Flatland 2D grid environment.

-----

## 📊 Project Overview & Status

  * **Question 1 & 2:** Solutions are complete and located in their respective files.
  * **Question 3:** The primary task is to develop a high-performing solution in `question3.py`.
      * An existing baseline solution with **69% performance** is backed up for reference in `org/question3_backup.py`.
      * The detailed performance metrics of the baseline can be found in `log.txt`.

-----

## 🎯 Goal

The main objective is to **implement and improve the solution for Question 3** by analyzing the existing code, optimizing algorithms, and referencing relevant research.

-----

## ⚠️ Important Guidelines

When working on the solution, the following rules are critical:

1.  All development must be done within the `question3.py` file.
2.  **Do not delete or modify any code sections that are explicitly marked with instructions not to be removed or changed.** Adhering to this constraint is mandatory.

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