Work Sample Problems - Data Role
===

## Work Sample

Apache Spark dockerized template and problem dataset: [`ws-data-spark`](https://github.com/EQWorks/ws-data-spark).

Preference will be given to solutions submitted as Spark Jobs, however, you are welcome to use other tools (such as [Pandas](https://pandas.pydata.org/)).

## Problems

### 1. Cleanup

A sample dataset of request logs is given in `data/DataSample.csv`. We consider records that have identical `geoinfo` and `timest` as suspicious. Please clean up the sample dataset by filtering out those suspicious request records.

### 2. Label

Assign each _request_ (from `data/DataSample.csv`) to the closest (i.e. minimum distance) _POI_ (from `data/POIList.csv`).

Note: a _POI_ is a geographical Point of Interest.

### 3. Analysis

1. For each _POI_, calculate the average and standard deviation of the distance between the _POI_ to each of its assigned _requests_.
2. At each _POI_, draw a circle (with the center at the POI) that includes all of its assigned _requests_. Calculate the radius and density (requests/area) for each _POI_.

### 4. Data Science/Engineering Tracks

Please complete either `4a` or `4b`. Extra points will be awarded for completing both tasks.

#### 4a. Model

1. To visualize the popularity of each _POI_, they need to be mapped to a scale that ranges from -10 to 10. Please provide a mathematical model to implement this, taking into consideration of extreme cases and outliers. Aim to be more sensitive around the average and provide as much visual differentiability as possible.
2. **Bonus**: Try to come up with some reasonable hypotheses regarding _POIs_, state all assumptions, testing steps and conclusions. Include this as a text file (with a name `bonus`) in your final submission.

#### 4b. Pipeline Dependency

We use a modular design on all of our data analysis tasks. To get to a final product, we organize steps using a data pipeline. One task may require the output of one or multiple other tasks to run successfully. This creates dependencies between tasks.

We also require the pipeline to be flexible. This means a new task may enter a running pipeline anytime that may not have the tasks' dependencies satisfied. In this event, we may have a set of tasks already running or completed in the pipeline, and we will need to map out which tasks are prerequisites for the newest task so the pipeline can execute them in the correct order. For optimal pipeline execution, when we map out the necessary tasks required to execute the new task, we want to avoid scheduling tasks that have already been executed.

If we treat each task as a node and the dependencies between a pair of tasks as directed edges, we can construct a DAG ([Wiki: Directed Acyclic Graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph)).

Consider the following scenario. At a certain stage of our data processing, we have a set of tasks (starting tasks) that we know all its prerequisite task has been executed, and we wish to reach to a later goal task. We need to map out a path that indicates the order of executions on tasks that finally leads to the goal task. We are looking for a solution that satisfies both necessity and sufficiency -- if a task is not a prerequisite task of goal, or its task is a prerequisite task for starting tasks (already been executed), then it shouldn't be included in the path. The path needs to follow a correct topological ordering of the DAG, hence a task needs to be placed behind all its necessary prerequisite tasks in the path.

Note: A starting task should be included in the path, if and only if it's a prerequisite of the goal task

For example, we have 6 tasks `[A, B, C, D, E, F]`, `C` depends on `A` (denoted as `A->C`), `B->C`, `C->E`, `E->F`. A new job has at least 2 tasks and at most 6 tasks, each task can only appear once.

Examples:

1. Inputs: starting task: `A`, goal task: `F`, output: `A,B,C,E,F` or `B,A,C,E,F`.
2. Input: starting task: `A,C`, goal task:'F', outputs: `C,E,F`.

You will find the starting task and the goal task in [`question.txt`](#file-question-txt) file, list of all tasks in [`task_ids.txt`](#file-relations-txt) and dependencies in [`relations.txt`](#file-task_ids-txt).

Please submit your implementation and result.

## Notes

Keep in mind that both the amount and the dimension of the data we work with are much higher in reality, so try to demonstrate that your solutions are capable of handling beyond the sample scale.

## Submission

- A git remote repository (GitHub, bitbucket, gitlab, your server, virtually anywhere) of your implementation.
- `git-archive` of your implementation. In case of some email providers blocking attachments, use Box.com, Dropbox, Google Drive, your FTP server, or anything that you have the easiest access.
