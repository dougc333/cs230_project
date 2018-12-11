duplicate question detection based on quora dataset

To make the datasets: 
Run ./datasets/getglove.bash this downloads the glove vectors
unzip and move into datasets/GloVe. There should be a glove.840B.300d.txt file
run the following in the make_data.ipynb jupyter notebook. This creates the training/validation/test data sets and 
stores them as pkl files. There are 2 datasets a small dataset to be used for faster debugging and a full dataset. 
The code in the cells looks like: 
# test create large dataset
X_train,X_valid,X_test,y_train,y_valid,y_test = make_dataset('/home/dc/cs230_project')
save(X_train_small,X_valid_small,X_test_small,y_train_small,y_valid_small,y_test_small)


X_train,X_valid,X_test,y_train,y_valid,y_test = load_data()
print("should see large dataset: 404xxx")
print(type(X_train),X_train.shape,type(y_train),y_train.shape)
print(type(X_valid),X_valid.shape,type(y_valid),y_valid.shape)
print(type(X_test),X_test.shape,type(y_test),y_test.shape)


X_train_small,X_valid_small,X_test_small,y_train_small,y_valid_small,y_test_small \
                = make_dataset('/home/dc/cs230_project',small=True)
save(X_train_small,X_valid_small,X_test_small,y_train_small,y_valid_small,y_test_small,small=True)

X_train,X_valid,X_test,y_train,y_valid,y_test = load_data(small=True)
print("should see small dataset: 404xxx")
print(type(X_train),X_train.shape,type(y_train),y_train.shape)
print(type(X_valid),X_valid.shape,type(y_valid),y_valid.shape)
print(type(X_test),X_test.shape,type(y_test),y_test.shape)


To run the training code: 
python train.py
This prints the train, test and validation errors. 

