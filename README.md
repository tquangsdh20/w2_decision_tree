# Bài Tập Decision Tree

## Use Decision Tree to classify the Iris dataset

- dataset: 'Iris.csv'


### Đọc dữ liệu

```python
import pandas as pd
dat = pd.read_csv('Iris.csv')
print(dat.shape)
```

### Mô tả dữ liệu

Bộ dữ liệu về hoa Iris được thu thập bởi Edgar Anderson, ông là một nhà thực vật học, sau đó Ronald Aylmer Fisher đã thống kê và rút gọn lại với các thuộc tính chính như sau:  
-	Sepal Length: Chiều dài của dài hoa
-	Sepal Width: Chiều rộng đài hoa
-	Petal Length: Chiều dài đài hoa
-	Petal Width: Chiều rộng cánh hoa
-	Nhãn với tên loại hoa gồm 3 loại *Iris Setora*, *Iris Verginica*, *Iris Versicolor*  

Ngoài ra tập dữ liệu về hoa Iris có tổng cộng 150 cá thể và phân bố đều cho cả ba loại hoa mỗi loại là 50 cá thể. Do đó, tập dữ liệu trên thuộc tập dữ liệu cân bằng với các thông số thông kê như bảng sau:  

## 1) Data Exploration


```python
dat.head()
```




<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
dat.describe()
```



<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>75.500000</td>
      <td>5.843333</td>
      <td>3.054000</td>
      <td>3.758667</td>
      <td>1.198667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>43.445368</td>
      <td>0.828066</td>
      <td>0.433594</td>
      <td>1.764420</td>
      <td>0.763161</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>38.250000</td>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>75.500000</td>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>112.750000</td>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>150.000000</td>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>



```python
dat.groupby('Species').size()
```




    Species
    Iris-setosa        50
    Iris-versicolor    50
    Iris-virginica     50
    dtype: int64



## 2) Prepare train dataset and test dataset


```python
# Get the X as all the features, and Y as the labels
X = dat.drop('Species', axis=1)  
y = dat['Species']
```



<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Slit the dataset into 2 datasets (80% / 20%)
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(f"The number of individuals in TRAIN dataset: {X_train.shape[0]}")
print(f"The number of individuals in TEST dataset :  {X_test.shape[0]}")
```

    The number of individuals in TRAIN dataset: 120
    The number of individuals in TEST dataset :  30
    


```python
## Missing Data Checking:
pd.isnull(X_train).any() | pd.isnull(X_test).any()
```




    Id               False
    SepalLengthCm    False
    SepalWidthCm     False
    PetalLengthCm    False
    PetalWidthCm     False
    dtype: bool



There's no missing data in the dataset

## Training Dataset with Decision Tree

### a) criterion='gini'

Building the Decision Tree with the **gini index**


```python
# Decision Tree Classifier is supported by the Scikit-Learn
from sklearn.tree import DecisionTreeClassifier  
dt = DecisionTreeClassifier(criterion='gini')  
dt.fit(X_train, y_train)  
```




    DecisionTreeClassifier()



#### Predict the test data


```python
y_pred_dt = dt.predict(X_test)  
print(y_pred_dt)
```

    ['Iris-versicolor' 'Iris-setosa' 'Iris-versicolor' 'Iris-versicolor'
     'Iris-versicolor' 'Iris-virginica' 'Iris-setosa' 'Iris-virginica'
     'Iris-virginica' 'Iris-versicolor' 'Iris-virginica' 'Iris-versicolor'
     'Iris-setosa' 'Iris-setosa' 'Iris-versicolor' 'Iris-setosa'
     'Iris-virginica' 'Iris-virginica' 'Iris-setosa' 'Iris-virginica'
     'Iris-virginica' 'Iris-versicolor' 'Iris-virginica' 'Iris-versicolor'
     'Iris-virginica' 'Iris-setosa' 'Iris-virginica' 'Iris-setosa'
     'Iris-virginica' 'Iris-virginica']
    

#### Evaluating the Algorithm


```python
dt_score = dt.score(X_test, y_test)
print(f"Decision Tree classifier accuracy score is {dt_score}")
```

    Decision Tree classifier accuracy score is 1.0
    

### b) criterion='entropy'

Building the tree with the **entropy**


```python
from sklearn.tree import DecisionTreeClassifier  
dt2 = DecisionTreeClassifier(criterion='entropy')  
dt2.fit(X_train, y_train)  
```




    DecisionTreeClassifier(criterion='entropy')




```python
y_pred_dt = dt.predict(X_test)  
print(y_pred_dt)
```

    ['Iris-versicolor' 'Iris-setosa' 'Iris-versicolor' 'Iris-versicolor'
     'Iris-versicolor' 'Iris-virginica' 'Iris-setosa' 'Iris-virginica'
     'Iris-virginica' 'Iris-versicolor' 'Iris-virginica' 'Iris-versicolor'
     'Iris-setosa' 'Iris-setosa' 'Iris-versicolor' 'Iris-setosa'
     'Iris-virginica' 'Iris-virginica' 'Iris-setosa' 'Iris-virginica'
     'Iris-virginica' 'Iris-versicolor' 'Iris-virginica' 'Iris-versicolor'
     'Iris-virginica' 'Iris-setosa' 'Iris-virginica' 'Iris-setosa'
     'Iris-virginica' 'Iris-virginica']
    

#### Evaluating the Algorithm


```python
dt_score = dt.score(X_test, y_test)
print(f"Decision Tree classifier accuracy score is {dt_score}")
```

    Decision Tree classifier accuracy score is 1.0
    

### Visualize decision tree


```python
from sklearn.tree import export_graphviz
dot_data = export_graphviz(dt, out_file=None)
print(dot_data)
```

    digraph Tree {
    node [shape=box] ;
    0 [label="X[4] <= 0.8\ngini = 0.666\nsamples = 120\nvalue = [42, 41, 37]"] ;
    1 [label="gini = 0.0\nsamples = 42\nvalue = [42, 0, 0]"] ;
    0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
    2 [label="X[0] <= 100.5\ngini = 0.499\nsamples = 78\nvalue = [0, 41, 37]"] ;
    0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
    3 [label="gini = 0.0\nsamples = 41\nvalue = [0, 41, 0]"] ;
    2 -> 3 ;
    4 [label="gini = 0.0\nsamples = 37\nvalue = [0, 0, 37]"] ;
    2 -> 4 ;
    }
    


```python

```
