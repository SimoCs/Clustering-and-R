# Clustering and Classifying Wine Using Machine Learning, R, and GridDB

This article will cover the creation of wine clusters based on a **Wine** dataset's different attributes. **R Programming** will be used, which is very useful in creating a set of groups representing some of the differences and similarities found in the types of wines. In addition, **GridDB** will be our central database for this program as it is ideally suited to hold machine learning datasets. The article will outline the requirements to set up our database **GridDB**. Following that, we will briefly describe our dataset and model. To finish off, we will interpret the results and come up with our conclusion.

## Requirements

The **GridDB** database storage system will store our dataset while building our clustering machine learning model. **GridDB** must be downloaded and configured in your operating system to be fully functional.

Make sure to run the following R statements to import the needed libraries helpful in running our **Wine** Cluster:


```R
install.packages( "RJDBC")
install.packages("factoextra")
```


```R
library(tidyverse)
library(RJDBC)
library(factoextra)
```

## The Dataset

To implement the clustering algorithm, we will use the **wine** dataset to create wine clusters. The dataset contains **14** attributes and **178** instances. 

To conduct this task, we will run the following code:


```R
data <- read_csv("data.csv", show_col_types = FALSE)
```

The attributes covered in this dataset are as follows:

- **Alcohol**  this attribute is a numeric value.
- **Malic Acid** this attribute is a numeric value.
- **Ash** this attribute is a numeric value.
- **Ash Alcanity** this attribute is a numeric value.
- **Magnesium** this attribute is a numeric value.
- **Total Phenols** this attribute is a numeric value.
- **Flavanoids** this attribute is a numeric value.
- **Nonflavanoid Phenols** this attribute is a numeric value.
- **Proanthocyanins** this attribute is a numeric value.
- **Color Intensity** this attribute is a numeric value.
- **Hue** this attribute is a numeric value.
- **OD280** this attribute is a numeric value.
- **Proline** this attribute is a numeric value.
- **Customer Segment** this attribute is a numeric value that takes **three** categories of segments **1**, **2**, and **3**.

Find below an extract of the dataset:


```R
glimpse(data)
```

    Rows: 178
    Columns: 14
    $ Alcohol              [3m[90m<dbl>[39m[23m 14.23, 13.20, 13.16, 14.37, 13.24, 14.20, 14.39, …
    $ Malic_Acid           [3m[90m<dbl>[39m[23m 1.71, 1.78, 2.36, 1.95, 2.59, 1.76, 1.87, 2.15, 1…
    $ Ash                  [3m[90m<dbl>[39m[23m 2.43, 2.14, 2.67, 2.50, 2.87, 2.45, 2.45, 2.61, 2…
    $ Ash_Alcanity         [3m[90m<dbl>[39m[23m 15.6, 11.2, 18.6, 16.8, 21.0, 15.2, 14.6, 17.6, 1…
    $ Magnesium            [3m[90m<dbl>[39m[23m 127, 100, 101, 113, 118, 112, 96, 121, 97, 98, 10…
    $ Total_Phenols        [3m[90m<dbl>[39m[23m 2.80, 2.65, 2.80, 3.85, 2.80, 3.27, 2.50, 2.60, 2…
    $ Flavanoids           [3m[90m<dbl>[39m[23m 3.06, 2.76, 3.24, 3.49, 2.69, 3.39, 2.52, 2.51, 2…
    $ Nonflavanoid_Phenols [3m[90m<dbl>[39m[23m 0.28, 0.26, 0.30, 0.24, 0.39, 0.34, 0.30, 0.31, 0…
    $ Proanthocyanins      [3m[90m<dbl>[39m[23m 2.29, 1.28, 2.81, 2.18, 1.82, 1.97, 1.98, 1.25, 1…
    $ Color_Intensity      [3m[90m<dbl>[39m[23m 5.64, 4.38, 5.68, 7.80, 4.32, 6.75, 5.25, 5.05, 5…
    $ Hue                  [3m[90m<dbl>[39m[23m 1.04, 1.05, 1.03, 0.86, 1.04, 1.05, 1.02, 1.06, 1…
    $ OD280                [3m[90m<dbl>[39m[23m 3.92, 3.40, 3.17, 3.45, 2.93, 2.85, 3.58, 3.58, 2…
    $ Proline              [3m[90m<dbl>[39m[23m 1065, 1050, 1185, 1480, 735, 1450, 1290, 1295, 10…
    $ Customer_Segment     [3m[90m<dbl>[39m[23m 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1…
    

To display the first **5** rows of our dataset, we can use the **head** method to do so:


```R
head(data, n = 5)
```


<table class="dataframe">
<caption>A tibble: 5 × 14</caption>
<thead>
	<tr><th scope=col>Alcohol</th><th scope=col>Malic_Acid</th><th scope=col>Ash</th><th scope=col>Ash_Alcanity</th><th scope=col>Magnesium</th><th scope=col>Total_Phenols</th><th scope=col>Flavanoids</th><th scope=col>Nonflavanoid_Phenols</th><th scope=col>Proanthocyanins</th><th scope=col>Color_Intensity</th><th scope=col>Hue</th><th scope=col>OD280</th><th scope=col>Proline</th><th scope=col>Customer_Segment</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>14.23</td><td>1.71</td><td>2.43</td><td>15.6</td><td>127</td><td>2.80</td><td>3.06</td><td>0.28</td><td>2.29</td><td>5.64</td><td>1.04</td><td>3.92</td><td>1065</td><td>1</td></tr>
	<tr><td>13.20</td><td>1.78</td><td>2.14</td><td>11.2</td><td>100</td><td>2.65</td><td>2.76</td><td>0.26</td><td>1.28</td><td>4.38</td><td>1.05</td><td>3.40</td><td>1050</td><td>1</td></tr>
	<tr><td>13.16</td><td>2.36</td><td>2.67</td><td>18.6</td><td>101</td><td>2.80</td><td>3.24</td><td>0.30</td><td>2.81</td><td>5.68</td><td>1.03</td><td>3.17</td><td>1185</td><td>1</td></tr>
	<tr><td>14.37</td><td>1.95</td><td>2.50</td><td>16.8</td><td>113</td><td>3.85</td><td>3.49</td><td>0.24</td><td>2.18</td><td>7.80</td><td>0.86</td><td>3.45</td><td>1480</td><td>1</td></tr>
	<tr><td>13.24</td><td>2.59</td><td>2.87</td><td>21.0</td><td>118</td><td>2.80</td><td>2.69</td><td>0.39</td><td>1.82</td><td>4.32</td><td>1.04</td><td>2.93</td><td> 735</td><td>1</td></tr>
</tbody>
</table>



## Data Analysis

Before storing or running our machine learning cluster, we will have to analyze the data. As a start, we will inspect the datatypes of our columns. 

To conduct this task, we will run the following code:


```R
str(data)
```

    spc_tbl_ [178 × 14] (S3: spec_tbl_df/tbl_df/tbl/data.frame)
     $ Alcohol             : num [1:178] 14.2 13.2 13.2 14.4 13.2 ...
     $ Malic_Acid          : num [1:178] 1.71 1.78 2.36 1.95 2.59 1.76 1.87 2.15 1.64 1.35 ...
     $ Ash                 : num [1:178] 2.43 2.14 2.67 2.5 2.87 2.45 2.45 2.61 2.17 2.27 ...
     $ Ash_Alcanity        : num [1:178] 15.6 11.2 18.6 16.8 21 15.2 14.6 17.6 14 16 ...
     $ Magnesium           : num [1:178] 127 100 101 113 118 112 96 121 97 98 ...
     $ Total_Phenols       : num [1:178] 2.8 2.65 2.8 3.85 2.8 3.27 2.5 2.6 2.8 2.98 ...
     $ Flavanoids          : num [1:178] 3.06 2.76 3.24 3.49 2.69 3.39 2.52 2.51 2.98 3.15 ...
     $ Nonflavanoid_Phenols: num [1:178] 0.28 0.26 0.3 0.24 0.39 0.34 0.3 0.31 0.29 0.22 ...
     $ Proanthocyanins     : num [1:178] 2.29 1.28 2.81 2.18 1.82 1.97 1.98 1.25 1.98 1.85 ...
     $ Color_Intensity     : num [1:178] 5.64 4.38 5.68 7.8 4.32 6.75 5.25 5.05 5.2 7.22 ...
     $ Hue                 : num [1:178] 1.04 1.05 1.03 0.86 1.04 1.05 1.02 1.06 1.08 1.01 ...
     $ OD280               : num [1:178] 3.92 3.4 3.17 3.45 2.93 2.85 3.58 3.58 2.85 3.55 ...
     $ Proline             : num [1:178] 1065 1050 1185 1480 735 ...
     $ Customer_Segment    : num [1:178] 1 1 1 1 1 1 1 1 1 1 ...
     - attr(*, "spec")=
      .. cols(
      ..   Alcohol = [32mcol_double()[39m,
      ..   Malic_Acid = [32mcol_double()[39m,
      ..   Ash = [32mcol_double()[39m,
      ..   Ash_Alcanity = [32mcol_double()[39m,
      ..   Magnesium = [32mcol_double()[39m,
      ..   Total_Phenols = [32mcol_double()[39m,
      ..   Flavanoids = [32mcol_double()[39m,
      ..   Nonflavanoid_Phenols = [32mcol_double()[39m,
      ..   Proanthocyanins = [32mcol_double()[39m,
      ..   Color_Intensity = [32mcol_double()[39m,
      ..   Hue = [32mcol_double()[39m,
      ..   OD280 = [32mcol_double()[39m,
      ..   Proline = [32mcol_double()[39m,
      ..   Customer_Segment = [32mcol_double()[39m
      .. )
     - attr(*, "problems")=<externalptr> 
    

The next step is to provide summary statistics for every column we have in the wine dataset using the following code:


```R
summary(data)
```


        Alcohol        Malic_Acid         Ash         Ash_Alcanity  
     Min.   :11.03   Min.   :0.740   Min.   :1.360   Min.   :10.60  
     1st Qu.:12.36   1st Qu.:1.603   1st Qu.:2.210   1st Qu.:17.20  
     Median :13.05   Median :1.865   Median :2.360   Median :19.50  
     Mean   :13.00   Mean   :2.336   Mean   :2.367   Mean   :19.49  
     3rd Qu.:13.68   3rd Qu.:3.083   3rd Qu.:2.558   3rd Qu.:21.50  
     Max.   :14.83   Max.   :5.800   Max.   :3.230   Max.   :30.00  
       Magnesium      Total_Phenols     Flavanoids    Nonflavanoid_Phenols
     Min.   : 70.00   Min.   :0.980   Min.   :0.340   Min.   :0.1300      
     1st Qu.: 88.00   1st Qu.:1.742   1st Qu.:1.205   1st Qu.:0.2700      
     Median : 98.00   Median :2.355   Median :2.135   Median :0.3400      
     Mean   : 99.74   Mean   :2.295   Mean   :2.029   Mean   :0.3619      
     3rd Qu.:107.00   3rd Qu.:2.800   3rd Qu.:2.875   3rd Qu.:0.4375      
     Max.   :162.00   Max.   :3.880   Max.   :5.080   Max.   :0.6600      
     Proanthocyanins Color_Intensity       Hue             OD280      
     Min.   :0.410   Min.   : 1.280   Min.   :0.4800   Min.   :1.270  
     1st Qu.:1.250   1st Qu.: 3.220   1st Qu.:0.7825   1st Qu.:1.938  
     Median :1.555   Median : 4.690   Median :0.9650   Median :2.780  
     Mean   :1.591   Mean   : 5.058   Mean   :0.9574   Mean   :2.612  
     3rd Qu.:1.950   3rd Qu.: 6.200   3rd Qu.:1.1200   3rd Qu.:3.170  
     Max.   :3.580   Max.   :13.000   Max.   :1.7100   Max.   :4.000  
        Proline       Customer_Segment
     Min.   : 278.0   Min.   :1.000   
     1st Qu.: 500.5   1st Qu.:1.000   
     Median : 673.5   Median :2.000   
     Mean   : 746.9   Mean   :1.938   
     3rd Qu.: 985.0   3rd Qu.:3.000   
     Max.   :1680.0   Max.   :3.000   


To understand the distribution of our dataset and determine if any patterns will enable us to create clusters, we will develop a set of histograms for every one of our columns. 

This task can be coded as follows:


```R
data %>% keep(is.numeric) %>% gather() %>% ggplot(aes(value)) + 
  facet_wrap(~ key, scales = "free") + geom_histogram(bins = 30)
```


    
![png](output_19_0.png)
    


Another viewpoint to take while analyzing the data is determining the number of customers based on our three segments. Based on our analysis, these values are close from a numerical perspective. For the first segment, we have a total of **59** rows. The following customer, referred to as number **2**, is the high proportion of our dataset with **79** rows. Last but not least, the final segment is totaled to a total number of **48** rows. The ratio of customer segments is evenly matched without any of the segments being exceptionally high compared to the others. 

The code that performs this task in R is as follows:


```R
data %>%
  group_by(Customer_Segment) %>%
  summarise(count = n())
```


<table class="dataframe">
<caption>A tibble: 3 × 2</caption>
<thead>
	<tr><th scope=col>Customer_Segment</th><th scope=col>count</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>1</td><td>59</td></tr>
	<tr><td>2</td><td>71</td></tr>
	<tr><td>3</td><td>48</td></tr>
</tbody>
</table>



## Setup GridDB

To set up the GridDB database, we will initialize the connection object by specifying the **driverClass** and **classPath** attributes.

The following task can be as follows:


```R
drv <- JDBC(driverClass = "com.toshiba.mwcloud.gs.sql.Driver", classPath = "/jdbc/bin/gridstore-jdbc.jar")
```

The second step is to set up the **IP** address and **port** with the credentials in mind. These values will depend on your configuration and credentials. 

The task can be done as follows:


```R
griddb <- dbConnect(drv, "jdbc:gs://182.00.0.54:20008/dockerGridDB/public","admin", "admin")
```

## Store Data in GridDB

In this section, we will store the data in our GridDB database. We will use the `CREATE` query to complete this task, creating a table representing your wine dataset.

The following code was used to conduct the task explained in this section:


```R
dbInsertTable <- function(conn, name, 
                          df, append = TRUE){
                          for (i in seq_len(nrow(df))) {
                            dbWriteTable(conn, name, df[i, ], append = append)
                            }
                          }
```

One additional step is to use the function created to add the table to our GridDB database:


```R
dbSendUpdate(griddb, paste(
  "CREATE TABLE IF NOT EXISTS wine_clusters", 
  "(alcohol FLOAT, 
    acid FLOAT, 
    ash FLOAT, 
    alcanity FLOAT, 
    magnesium FLOAT, 
    phenols FLOAT,
    flavanoids FLOAT,
    nonflavanoid FLOAT,
    proanthocyanins FLOAT,
    colors FLOAT,
    hue FLOAT,
    od280 FLOAT,
    proline FLOAT,
    segment INTEGER);"))
```

The last step is to populate the table using the wine dataset:


```R
dbInsertTable(griddb, "wine_clusters", read_csv("data.csv"))
```

# Retrieve the Data from GridDB

In this section, we will retrieve the data from our database. We will use the `SELECT` query to complete this task, which returns all database values.

The following code was used to conduct the task explained in this section:


```R
queryString <- "select alcohol, acid, alcanity, magnesium, phenols, 
                flavanoids FLOAT, colors FLOAT, segment from wine_clusters"
```

To retrieve the data from our GridDB  database, we can use the **dbGetQuery** method to do so:


```R
data <- dbGetQuery(conn, queryString)
```

## Implementing a K-means Clustering in R

This article will employ a k-means cluster. To explain, this unsupervised machine learning model creates groups of attributes based on similarities in a dataset. To simplify, our model will create wine groups based on acidity, flavor, and other chemical alcohol features. 

To run a K-means model using R programming, we will have first to normalize our data frame using the `scale` method as follows:


```R
df_normalize <- as.data.frame(scale(data))
```

# Build the K-means Clustering

The K-means Clustering model will output the different wine groups based on their chemical similarities in our R program. Our code will first initialize the K-means Cluster object. The K-means Clustering algorithm will then be set up using the standard parameters.


```R
km.res <- kmeans(df_normalize, 3, nstart = 25)
```

# Conclusion & Results

The final section of our article is the understanding of our results. To easily digest how well our K-means Cluster successfully outputted the wine groups based on their chemical analysis. The main number we shall focus on in this section is the number of clusters. Which represents the total number of wine groups discovered using our K means algorithm. Our K-means Cluster outputted three groups. To explain, this is considered a great result as we can identify the edges of each group and their dimensions from the graph. For future development, our model will fully use the **GridDB** database as it provides an easy input-output data interface and an incredible speed of data retrieval.

The following cluster plot is a representation of our results:


```R
fviz_cluster(km.res, data = df_normalize, ellipse.type = "convex")
```


    
![png](output_47_0.png)
    

