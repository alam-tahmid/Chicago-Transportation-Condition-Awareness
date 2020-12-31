from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from operator import add
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.sql.functions import desc
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import row_number, monotonically_increasing_id
from pyspark.sql import Window
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import StandardScaler



spark = SparkSession.builder.appName("KmeansChicago").getOrCreate()
data = spark.read.csv('cc.csv', header='true')# spark dataframe

#data.show()#spark dataframe

acc_data_clus = data[
'POSTED_SPEED_LIMIT',
'TRAFFIC_CONTROL_DEVICE',
'DEVICE_CONDITION',
'WEATHER_CONDITION',
'LIGHTING_CONDITION',
'FIRST_CRASH_TYPE',
'TRAFFICWAY_TYPE',
'ALIGNMENT',
'ROADWAY_SURFACE_COND',
'ROAD_DEFECT',
'PRIM_CONTRIBUTORY_CAUSE',
'SEC_CONTRIBUTORY_CAUSE',
'CRASH_HOUR',
'CRASH_DAY_OF_WEEK',
'CRASH_MONTH',
'LATITUDE',
'LONGITUDE'
]

#acc_data_clus.show()

acc_data_clus.count()

acc_data_clus = acc_data_clus.filter(acc_data_clus.LATITUDE. isNotNull())

acc_data_clus.count()


indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(acc_data_clus) for column in list(set(acc_data_clus.columns))]


pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(acc_data_clus).transform(acc_data_clus)

#df_r.show()

df_r = df_r.withColumn(
    "index",
    row_number().over(Window.orderBy(monotonically_increasing_id()))-1
)

#df_r.show(100, False)
#df_r.printSchema()

#df_r.columns

trainData = df_r['WEATHER_CONDITION_index',
 'FIRST_CRASH_TYPE_index',
 'CRASH_HOUR_index',
 'CRASH_MONTH_index',
 'ROAD_DEFECT_index',
 'TRAFFIC_CONTROL_DEVICE_index',
 'POSTED_SPEED_LIMIT_index',
 'SEC_CONTRIBUTORY_CAUSE_index',
 'ALIGNMENT_index',
 'LATITUDE_index',
 'LIGHTING_CONDITION_index',
 'LONGITUDE_index',
 'ROADWAY_SURFACE_COND_index',
 'PRIM_CONTRIBUTORY_CAUSE_index',
 'CRASH_DAY_OF_WEEK_index',
 'TRAFFICWAY_TYPE_index',
 'DEVICE_CONDITION_index']

#trainData.show(5,False)
 
vecAssembler = VectorAssembler(inputCols=['WEATHER_CONDITION_index',
 'FIRST_CRASH_TYPE_index',
 'CRASH_HOUR_index',
 'CRASH_MONTH_index',
 'ROAD_DEFECT_index',
 'TRAFFIC_CONTROL_DEVICE_index',
 'POSTED_SPEED_LIMIT_index',
 'SEC_CONTRIBUTORY_CAUSE_index',
 'ALIGNMENT_index',
 'LATITUDE_index',
 'LIGHTING_CONDITION_index',
 'LONGITUDE_index',
 'ROADWAY_SURFACE_COND_index',
 'PRIM_CONTRIBUTORY_CAUSE_index',
 'CRASH_DAY_OF_WEEK_index',
 'TRAFFICWAY_TYPE_index',
 'DEVICE_CONDITION_index'], outputCol="to_features")
new_df = vecAssembler.transform(trainData)

#new_df.show()

scaler = StandardScaler(inputCol="to_features", outputCol="features",
                        withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(new_df)

# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(new_df)

#scaledData.printSchema()

#scaledData.select('features').show()

kmeans = KMeans(k=9, seed=1)  # 2 clusters here
model = kmeans.fit(scaledData.select('features'))

wssse = model.computeCost(scaledData)
print(wssse)

transformed = model.transform(scaledData)
#transformed.show()

centers = model.clusterCenters()
print(centers)

transformed.groupby('prediction').count().sort(desc('count')).show()

transformed_with_index = transformed.withColumn(
    "index",
    row_number().over(Window.orderBy(monotonically_increasing_id()))-1)

final_df = df_r.join(transformed_with_index,[df_r.index == transformed_with_index.index])


###################################################################################################################################################################################################
#final_df.printSchema()

#final_df.groupby('CRASH_MONTH','DEVICE_CONDITION','prediction').count().sort(desc('count')).show()
#final_df.groupby('CRASH_MONTH','WEATHER_CONDITION','POSTED_SPEED_LIMIT','prediction').count().sort(desc('count')).show()
#final_df.groupby('CRASH_HOUR','DEVICE_CONDITION','prediction').count().sort(desc('count')).show()
#final_df.groupby('PRIM_CONTRIBUTORY_CAUSE','LONGITUDE','LATITUDE','prediction').count().sort(desc('count')).show()
#final_df.groupby('CRASH_DAY_OF_WEEK','CRASH_HOUR','WEATHER_CONDITION','ROADWAY_SURFACE_COND','TRAFFIC_CONTROL_DEVICE','DEVICE_CONDITION','prediction').count().sort(desc('count')).show()
#final_df.groupby('CRASH_MONTH','prediction').count().sort(desc('count')).show()
###################################################################################################################################################################################################


final_df_df = final_df[
'POSTED_SPEED_LIMIT',
'TRAFFIC_CONTROL_DEVICE',
'DEVICE_CONDITION',
'WEATHER_CONDITION',
'LIGHTING_CONDITION',
'FIRST_CRASH_TYPE',
'TRAFFICWAY_TYPE',
'ALIGNMENT',
'ROADWAY_SURFACE_COND',
'ROAD_DEFECT',
'PRIM_CONTRIBUTORY_CAUSE',
'SEC_CONTRIBUTORY_CAUSE',
'CRASH_HOUR',
'CRASH_DAY_OF_WEEK',
'CRASH_MONTH',
'LATITUDE',
'LONGITUDE',
'prediction'
]
#final_df_df.groupby('prediction').count().sort(desc('count')).show()

final_df_df.write.format('com.databricks.spark.csv').save('mycsv', header='true')# Saving the final result

