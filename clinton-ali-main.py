import pyspark
import os
from graphframes import *
from pyspark.sql.functions import *
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType 
from pyspark.sql.window import Window

def findPrerequisiteTask(g, task, visited):
    """
    Returns the prerequisite tasks of the current task id.

    Keyword arguments:
    g -- GraphFrames object with id,dst,src
    task -- list of string of length 1 that contains the current task id
    visited -- list of string that contains previously visited vertices
    """
    sol = []
    prerequisite_task = g.shortestPaths(task)
    prerequisite_task = prerequisite_task.sort(["id"]).select("id", "distances").select("id", explode("distances")).orderBy("value",ascending=True)
    prerequisite_task = prerequisite_task.filter(~prerequisite_task.id.isin(visited)).select("id").filter(
        prerequisite_task.value > 0).select("id").rdd.flatMap(lambda x: x).collect()
    for pre_req_task in prerequisite_task:
        if pre_req_task not in visited:
            temp_sol, temp_visited = findPrerequisiteTask(g, [pre_req_task], visited + [pre_req_task])
            sol = temp_sol + sol
            visited = temp_sol + visited
    visited = task + visited
    sol = sol + task  
    return sol, visited


def findInitialVisited(g, task):
    """
    Returns the initial visited nodes to help filter out the results in the recursion.

    Keyword arguments:
    g -- GraphFrames object with id,dst,src
    task -- list of string of length 1 that contains the current task id
    visited -- list of string that contains previously visited vertices
    """
    visited = []
    prerequisite_task = g.shortestPaths(task)
    prerequisite_task = prerequisite_task.sort(["id"]).select("id", "distances").select("id", explode("distances")).orderBy("value",ascending=True)
    prerequisite_task = prerequisite_task.filter(prerequisite_task.value > 0).select("id").rdd.flatMap(lambda x: x).collect()
    visited = visited + task + prerequisite_task
    return visited

def readQuestion(pathToFile):
    """
    Reads in a .txt file and returns a list  of length 2 in the form of [[start_task_id], end_task_id]
    
    Example:
    questions.txt
    starting task: 73
    goal task: 36
    -> returns [[73],36]
    """
    output = []
    with open(pathToFile, "r") as file:
        for cnt, line in enumerate(file):
            targetId = line.split(": ")[1]
            targetId = targetId.strip().split(",")
            output.append(targetId)
    return output


def readTaskId(pathToFile):
    """
    Reads in a .txt file of a single line of task ids separated by commas. Returns the task ids as a list.
    Returns a list of tuples that represents the vertices of a graph.
    """
    with open(pathToFile, "r") as file:
        content = file.read()
    out = content.split(",")
    out = list(map(str,out))
    out = [(x,) for x in out]
    return out
taskId = readTaskId('task_ids.txt')

def readRelations(pathToFile):
    """
    Reads in a .txt file that represents the relationships between the task ids. They are delimited by '->'.
    Returns a list of tuples that represents directed edges of a graph.
    """
    with open(pathToFile, "r") as file:
        content = file.read()
    out = content.splitlines()
    out = [i.split('->', 1) for i in out]
    out = list(map(tuple, out)) 
    return out


if __name__ == '__main__':
    SUBMIT_ARGS = "--packages graphframes:graphframes:0.8.1-spark3.0-s_2.12 pyspark-shell"
    os.environ["PYSPARK_SUBMIT_ARGS"] = SUBMIT_ARGS
    
    conf = pyspark.SparkConf("local[4]")
    sc = pyspark.SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.appName('eqWorksAssessment').getOrCreate()

    # set schema for incoming data
    requests_schema = StructType([
        StructField("ID", IntegerType(), True),
        StructField("TimeStamp", StringType(), True),
        StructField("Country", StringType(), True),
        StructField("Province", StringType(), True),
        StructField("City", StringType(), True),
        StructField("Latitude", DoubleType(), True),
        StructField("Longitude", DoubleType(), True)])
    poi_schema = StructType([
        StructField("POIID", StringType(), True),
        StructField("poi_lat", DoubleType(), True),
        StructField("poi_long", DoubleType(), True)])

    dataSample = spark.read.csv("data/DataSample.csv",header=True,schema=requests_schema).coalesce(12)
    poi = spark.read.csv("data/POIList.csv",header=True,schema=poi_schema).cache()
    poi.registerTempTable('poi')
    target_cols = ["TimeStamp","Country","Province","City","Latitude","Longitude"]
    requests = dataSample.dropDuplicates(target_cols).coalesce(12).cache()

    print("Question 1.")
    print("Count of data sample before removing duplicates: " + str(dataSample.count()))
    print("Count of data sample after removing duplicates: " + str(requests.count()))

    print("Question 2")
    print("Since we later need to 'draw' a circle, I have chosen to implement the euclidean distance over the haversine distance.")
    # Resource: https://stackoverflow.com/questions/60086180/pyspark-how-to-apply-a-python-udf-to-pyspark-dataframe-columns
    combined = requests.crossJoin(poi).cache()
    combined = combined.withColumn("longitude_part", (col("poi_long") - col("Longitude")) ** 2) \
    .withColumn("latitude_part", (col("poi_lat") - col("Latitude")) ** 2) \
    .withColumn("euclidean_distance", sqrt(col("latitude_part") + col("longitude_part"))) \
    .drop("longitude_part", "latitude_part") # Implementation of euclidean distance
    w = Window.partitionBy(["ID","TimeStamp","Country","Province","City","Latitude","Longitude"]).orderBy('euclidean_distance')
    combined = combined.withColumn("rn", row_number().over(w)).where(col('rn') == 1).drop("rn").cache() # Select the minimum distance and assign to a POI
    combined = combined.dropDuplicates(target_cols).coalesce(12).cache()
    combined.show()
    combined.registerTempTable('combined')
    print("Check: There are " + str(combined.count()) + " entries in this dataframe. Then it means we still don't have duplicates after the cross join.")

    # Question 3
    print("Question 3")
    print("The average and standard deviation between the POI to each of its assigned requests is as follows.")
    print("Note that POI1 and POI2 are in the same exact location, and in this run POI2 have no requests assigned to it and have null as values.")
    print("Assumption: Given a POI with duplicate geographical data points, all requests will route to one of the POI with that data points")
    query3 = "SELECT poi.POIID, average_distance, stddev_samp_distance, radius, density FROM (SELECT POIID,\
            avg(euclidean_distance) as average_distance,\
            stddev_samp(euclidean_distance) as stddev_samp_distance,\
            max(euclidean_distance) as radius,\
            count(ID)/pow(max(euclidean_distance)*pi(),2) as density\
            FROM combined GROUP BY POIID) as combined RIGHT JOIN poi ON poi.POIID = combined.POIID"
    q3 = sqlContext.sql(query3)
    q3.show()

    print("Question 4a #1")
    print("Providing a mathematical model to map the popularity of POIs in a scale of [-10,10].")
    print("This solution is inspired by the boxplot way of visualizing data where we can see the centrality of the data based on the median which is not sensitive to outliers.")
    print("In essence we have two calculated fields which is the count of cities over province and density for each POI.")
    print("Number of countries was ommited because I the data comes from Canada only (Validated in bonus.txt).")
    print("Then we calculate the percentile rank over each of the rows. The percentile method will take care of the new requests coming in, that can be outliers, since it depends on the data itself and will scale accordingly.")
    print("We sum the percentiles of the calculated field (each calculated field have ranges [0,1]), and we get a data value of range [0,2]. We multiply by 10 and subtract by 10 to get the desired scale of [-10,10].")
    print("Note that POI2 is considered as popularity -10 as expected since no requests are ever routed there in this iteration.")
    print("Equations:")
    print("city_over_province = count(city)/count(province)")
    print("density = requests / area")
    print("Note: In order to scale this process perhaps recalculating the popularity over a period of time daily or 4x a day depending on the density of requests of the time and day.")
    print("popularity = (percent_rank(city_over_province) + percent_rank(density))*10 - 10")
    # Link to inspiring idea: https://www.sqlshack.com/calculate-sql-percentile-using-the-sql-server-percent_rank-function/")
    query4a1 = "SELECT * ,\
                (city_over_province_pct_rank + density_pct_rank)*10 - 10 as popularity \
                FROM (SELECT POIID,\
                    PERCENT_RANK()\
                        OVER(ORDER BY count_city_over_province) AS city_over_province_pct_rank,\
                    PERCENT_RANK()\
                        OVER(ORDER BY density) AS density_pct_rank\
                    FROM (SELECT poi.POIID, count_city_over_province, density FROM\
                        (SELECT POIID,\
                            COUNT(Distinct City)/Count(Distinct Province) as count_city_over_province, \
                            count(ID)/pow(max(euclidean_distance)*pi(),2) as density\
                        FROM combined GROUP BY POIID) as combined RIGHT JOIN poi ON poi.POIID = combined.POIID) as combined)"
    q4a1 = sqlContext.sql(query4a1)
    q4a1.show()

    # Need to find the minimum amount of tasks that are ordered based on the dag's topology.
    # We would do a postorder traversal when printing the tasks. I.e. dependent task then parent task
    # 
    print("Question 4b")
    print("We are using the graphframes package that is better suited to work with graphs.")
    print("Need to find the minimum amount of tasks that are ordered based on the dag's topology.")
    print("We begin by finding the shortest path from task 73 to task 36.")
    print("Then we would do a postorder traversal when printing the tasks. I.e. dependent task then parent task.")
    questionTask = readQuestion("data/question.txt")
    relations = readRelations('data/relations.txt')
    taskId = readTaskId('data/task_ids.txt')  
    vertices = sqlContext.createDataFrame(taskId,['id'])
    edges = sqlContext.createDataFrame(relations,['src','dst'])
    g = GraphFrame(vertices,edges).cache()
    g = g.dropIsolatedVertices().cache()
    candidate = g.shortestPaths(questionTask[1])
    # sort by shortest path first we will begin with that
    candidate = candidate.select("id", "distances").select("id", explode("distances")).filter(candidate.id.isin(questionTask[0])).orderBy("value",ascending=True)
    candidate_first_task = candidate.first().id
    minimum_path = g.bfs("id='"+candidate_first_task+"'","id='"+questionTask[1][0]+"'")
    minimum_path = sc.parallelize(minimum_path.select("from.id","v1.id","v2.id","to.id").first())
    minimum_path = minimum_path.collect()
    print("This is our shortest path from the task start to task end: " + str(minimum_path))
    print("We need to find all the dependent tasks in between each task excluding the task start.")

    sol = []
    visited = []
    visited = findInitialVisited(g,[candidate_first_task])
    sol = sol + [candidate_first_task]
    for i in range(1, len(minimum_path)):
        new_sol, new_visited = findPrerequisiteTask(g,[minimum_path[i]],visited)
        sol = sol + new_sol
        visited = new_visited
    print("These are all the tasks needed to be completed starting from from start to end: " + str(sol))
    sc.stop()