# this works
# from boto3 import client
# from io import BytesIO
from pandas import read_csv, DataFrame, isna
#import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# saving artifacts
import tempfile
import joblib
import pickle

# classifiers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve,roc_auc_score, auc,f1_score, accuracy_score, cohen_kappa_score, plot_confusion_matrix,precision_recall_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier

# ================================================ FILES =================================================================            

def csv(bucket, file_location):
    '''
    '''
    s3 = client('s3')

    # bucket='adl-core-sagemaker-studio'
    # key = 'external/Harish/claims_zz_obj_Harish_MH_2021_10_27.csv'
    obj = s3.get_object(Bucket=bucket, Key=file_location)
    df = read_csv(BytesIO(obj['Body'].read()))
    return df

def file_time():
    '''
    To provide uqniue file name ending based on time
    '''
    return str(datetime.now()).split(".")[0].replace("-","").replace(" ","").replace(":","")
# ================================================ EDA =================================================================            

def show_value_counts(table):
    for i in table.columns:
        print(f"**************************************************** {i}")
        print(table[i].value_counts())
        
def show_nans(table):
    '''
    Shows missing values count and percentage from highest to lowest
    '''
    print("This is only nans. Unknown, outliers, unfit values are not accounted for here\n\n")
    nan_count = DataFrame(table.isna().sum(), columns = ["actual_nan"])
    nan_count["percentage_missing"] = nan_count["actual_nan"]/table.shape[0]
    nan_count["column"] = nan_count.index
    nan_count.index = range(nan_count.shape[0])
    return nan_count.sort_values("percentage_missing",ascending=False)[["column","actual_nan","percentage_missing"]]
        
def show_histogram(table):
    '''
    plots relevant column's histogram 
    '''
    for i in table.columns:
        if(table[i].dtype!=object):
            print(i)
            plt.hist(table[i])
            plt.show()
            
def show_boxplot(table):
    '''
    plots relevant column's boxplot
    '''
    filtered_table = table[~isna(table)]
    for i in table.columns:
        if(filtered_table[i].dtype!=object):
            print(i)
            plt.boxplot(filtered_table[i])
            plt.show()

def show_roc_curve(y_test,pred, label):
    '''
    plot relevant roc curve
    '''
    # plot no skill roc curve
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # calculate roc curve for model
    fpr, tpr, _ = roc_curve(y_test, pred)
    # plot model roc curve
    plt.plot(fpr, tpr, marker='.', label=label)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
def show_aucpr_curve(y_test,pred,label):
    '''
    plot relevant aucpr curve
    '''

    # calculate the no skill line as the proportion of the positive class
    no_skill = len(y_test[y_test['target']==1]) / len(y_test)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # plot model precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, pred)
    plt.plot(recall, precision, marker='.', label=label)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


# ================================================ MODEL BUILDING =================================================================     

def build_default_classifier(df, test_size, target_column, classes):
    '''
    To pass paramters and get back metrics for all the defualt models
    No SVM right now
    '''
    # TRAIN TEST VALIDATION SPLIT
    X = df.drop(target_column,axis=1)
    y = df[target_column] 
    X_train, X_test, y_train, y_test = train_test_split(X ,y ,test_size=test_size, shuffle=True, random_state=123)
    print(f"Training : {X_train.shape}\nTest : {X_test.shape}" )
    metrics_keeper = DataFrame()
    
    lr_model_1 = LogisticRegression()
    rf_model_1 = RandomForestClassifier()
    dt_model_1 = DecisionTreeClassifier()
    nb_model_1 = GaussianNB()
    knn_model_1 = KNeighborsClassifier()
    sgd_model_1 = SGDClassifier()
    gb_model_1 = GradientBoostingClassifier()
    
    # Logistic
    lr_clf = lr_model_1.fit(X_train, y_train)
    model_name = "lr_model_1"
    pred  = lr_clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    kappa = cohen_kappa_score(y_test, pred)
    roc = roc_auc_score(y_test, pred)
    clr = [i for i in classification_report(y_test, pred).split(" ") if i !="" and "\n" not in i][:-9]
    metrics_keeper = better_report(accuracy, kappa, roc, classes, clr, model_name).append(metrics_keeper)
    print(f"1/7 Logistic regression DONE\n{'*'*90}")
    
    # Random forest
    rf_clf = rf_model_1.fit(X_train, y_train)
    model_name = "rf_model_1"
    pred  = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    kappa = cohen_kappa_score(y_test, pred)
    roc = roc_auc_score(y_test, pred)
    clr = [i for i in classification_report(y_test, pred).split(" ") if i !="" and "\n" not in i][:-9]
    metrics_keeper = better_report(accuracy, kappa, roc, classes, clr, model_name).append(metrics_keeper)
    print(f"2/7 Random Forest DONE\n{'*'*90}")
    
    # Decision tree
    dt_clf = dt_model_1.fit(X_train, y_train)
    model_name = "dt_model_1"
    pred  = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    kappa = cohen_kappa_score(y_test, pred)
    roc = roc_auc_score(y_test, pred)
    clr = [i for i in classification_report(y_test, pred).split(" ") if i !="" and "\n" not in i][:-9]
    metrics_keeper = better_report(accuracy, kappa, roc, classes, clr, model_name).append(metrics_keeper)
    print(f"3/7 Decision Tress DONE\n{'*'*90}")
    
    # naive bayes
    nb_clf = nb_model_1.fit(X_train, y_train)
    model_name = "nb_model_1"
    pred  = nb_clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    kappa = cohen_kappa_score(y_test, pred)
    roc = roc_auc_score(y_test, pred)
    clr = [i for i in classification_report(y_test, pred).split(" ") if i !="" and "\n" not in i][:-9]
    metrics_keeper = better_report(accuracy, kappa, roc, classes, clr, model_name).append(metrics_keeper)
    print(f"4/7 Navie Bayes DONE\n{'*'*90}")
    
    # k nearest
    knn_clf = knn_model_1.fit(X_train, y_train)
    model_name = "knn_model_1"
    pred  = knn_clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    kappa = cohen_kappa_score(y_test, pred)
    roc = roc_auc_score(y_test, pred)
    clr = [i for i in classification_report(y_test, pred).split(" ") if i !="" and "\n" not in i][:-9]
    metrics_keeper = better_report(accuracy, kappa, roc, classes, clr, model_name).append(metrics_keeper)
    print(f"5/7 Nearest Neighbour DONE\n{'*'*90}")
    
    # Gradient boosting 
    gb_clf = gb_model_1.fit(X_train, y_train)
    model_name = "gb_model_1"
    pred  = gb_clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    kappa = cohen_kappa_score(y_test, pred)
    roc = roc_auc_score(y_test, pred)
    clr = [i for i in classification_report(y_test, pred).split(" ") if i !="" and "\n" not in i][:-9]
    metrics_keeper = better_report(accuracy, kappa, roc, classes, clr, model_name).append(metrics_keeper)
    print(f"6/7 Gradient Boost DONE\n{'*'*90}")
    
    # Stocastic gradient 
    sgd_clf = sgd_model_1.fit(X_train, y_train)
    model_name = "sgd_model_1"
    pred  = sgd_clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    kappa = cohen_kappa_score(y_test, pred)
    roc = roc_auc_score(y_test, pred)
    clr = [i for i in classification_report(y_test, pred).split(" ") if i !="" and "\n" not in i][:-9]
    metrics_keeper = better_report(accuracy, kappa, roc, classes, clr, model_name).append(metrics_keeper)
    print(f"7/7 Stochastic gradient descent DONE\n{'*'*90}")
    
    metrics_keeper.index = range(metrics_keeper.shape[0])
    return {"metrics" : metrics_keeper, "logistic" : lr_model_1, "random forest":rf_model_1, "decision":dt_model_1, "naive":nb_model_1, "nearest":knn_model_1, "stochastic":sgd_model_1, "gradient":gb_model_1}



def better_report(accuracy, kappa, roc, aucpr, classes, clr, model_name):
    '''
    Given the model details, a dataframe is retuned based on classification report but better looking
    '''
    prf = clr[:3] # precision, recall, f1
    prf_scores = clr[3:-2] 
    prf_classes = prf_scores[::4] # labels
    prf_classes_scores = [i for i in prf_scores if i not in prf_classes]
    final_scores = [prf_classes_scores[i:i + len(prf)] for i in range(0, len(prf_classes_scores), len(prf))] # scores dvivded by laebl
    metric_df = DataFrame(final_scores, columns=prf)
    metric_df["label"] = prf_classes
    metric_df["accuracy"] = accuracy
    metric_df["kappa"] = kappa
    metric_df["auc-roc"] = roc
    metric_df["aucpr"] = aucpr
    metric_df["model"] = model_name
    return metric_df

def find_best_model(metric_df, lesser_label, overfitting_threshold):
    '''
    1. Anything above a threshold passed will be ignored
    2. 
    '''
    return metric_df[(metric_df["label"] == lesser_label) & (metric_df["accuracy"] < overfitting_threshold)].sort_values(["kappa","aucpr","auc-roc",'f1-score',"accuracy","recall","precision"], ascending=False).head(3)["model"].values

# ================================================ SAVING MODELS =================================================================       

def save_to_s3(s3_client, model, bucket, key):
    '''
    '''
    with tempfile.TemporaryFile() as fp:
        joblib.dump(model, fp)
        fp.seek(0)
        s3_client.put_object(Body=fp.read(), Bucket=bucket, Key=key)

def read_from_s3(s3_client, bucket, key):
    '''
    '''
    with tempfile.TemporaryFile() as fp:
        s3_client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
        fp.seek(0)
        model = joblib.load(fp)


# ================================================ CLEANING COLUMNS =================================================================            
            
def check_for_not_found(column, mapper, column_name):
    '''
    To check if there are values which are not recorded and might need attention while cleaning
    '''
    unmatched = [i for i in column.unique() if i not in mapper.keys()]
    if(len(unmatched)>0):
        print(f"!!! These were not found for {column_name} in the mapper : {unmatched}")
    else:
        print(f"Mapping was successful for : {column_name}")

        
def clean_gender(column):
    '''
    M F Other Unknown
    1 2 3     0
    '''
    mapper = {
    "M":"M",\
    "F":"F",\
    "Female" : "F",\
    "Male" : "M",\
    "female" : "F",\
    "male" : "M",\
    "U":"U",\
    "Unknown":"U",\
    "Not specified":"U",\
    "N":"U",\
    "Not declared":"U",\
    "Decline to answer":"U",\
    "Gender_Female" : "F",\
    "Gender_Male" : "M",\
    "O":"O",\
    "Non Binary":"O",\
    "Undeclared":"U",\
    "D":"U",\
    "Not_declared":"U",\
    "C":"U",\
    "I choose not to disclose":"U",\
    "Choose not to Disclose":"U",\
    'Declined to State': "U",\
    'Choose not to disclose': "U",\
    'I Choose not to Disclose': "U",\
    'Not Listed': "U",\
    'I do not wish to provide this information':"U",\
    'Not Declared': "U",\
    'ND': "U",\
    np.nan:"U"
                             }
    check_for_not_found(column, mapper, "gender")
    return column.map(mapper)

def clean_marital_status(column):
    '''
    Too many unknowns, needs update
    S M D W Unknown
    1 2 3 4 0
    '''
    mapper = {
    'M':"M",\
    'S':"S",\
    np.nan:"U",\
    'Married_United_States_of_America':"M",\
    'Single_United_States_of_America':"S",\
    'Single':"S",\
    'USA_Single':"S",\
    'Divorced_United_States_of_America':"D",\
    'Married USA':"M",\
    'M-USA':"M",\
    'Married_USA':"M",\
    'S-USA':"S",\
    'Unknown_United_States_of_America':"U",\
    'Married':"M",\
    'Divorced USA':"D",\
    'Not_Indicated_United_States_of_America':"U",\
    'Single USA':"S",\
    'Partnered_United_States_of_America':"M",\
    'Divorced_USA':"D",\
    'Divorced':"D",\
    'Single_USA':"S",\
    'P-USA':"U",\
    'Widowed_United_States_of_America':"W",\
    'USA-Single':"S",\
    'DE_FACTO':"U",\
    'Not_Disclosed_United_States_of_America':"U",\
    'USA_Divorced':"D",\
    'USA-Unknown':"U",\
    'Single _United_States_of_America_Japan':"S",\
    'USA-Married':"M",\
    'D-USA':"D",\
    'Separated_United_States_of_America':"D",\
    'USA_Married':"M",\
    'Co-Habiting_United_States_of_America':"M",\
    'Married_United_States_of_America_Japan':"M",\
    'U-USA':"U",\
    'RDP':"U",\
    'MI_NOT_DISCLOSED':"U",\
    'Separated USA':"D",\
    'Married_Ireland':"M",\
    'Common_Law_United_States_of_America':"U",\
    'Civil_Partnership_United_States_of_America':"M",\
    'Widowed_USA':"W",\
    'USA_Widowed':"W",\
    'MARITAL_STATUS-6-302':"U",\
    'Widowed USA':"W",\
    'Registered_Partnership_United_States_of_America':"M",\
    'Legally_Separated_United_States_of_America':"D",\
    'USA-Divorced':"D",\
    'M-CAN':"M",\
    'MARITAL_STATUS-3-40':"U",\
    'W-USA':"W",\
    'Married_Switzerland':"M",\
    'Unknown_USA':"U",\
    'O-USA':"U",\
    'M-IND':"M",\
    'Partnered USA':"M",\
    'USA_Living together':"M",\
    'Married _United_States_of_America_Italy':"M",\
    'Head_of_Household_USA':"M",\
    'MARITAL_STATUS-3-69':"U",\
    'R-USA':"U",\
    'MARITAL_STATUS-3-323':"U",\
    'USA-Married/ Civil Partnership':"M",\
    'Domestic Partner':"M",\
    'L-USA':"U",\
    'S-IND':"S",\
    'Hd Hsehld_United_States_of_America':"M",\
    'Living_Together_United_States_of_America':"M",\
    'M-HKG':"M",\
    'S-GBR':"S",\
    'USA_Married/Civil Partnership':"M",\
    'USA-Cohabit':"M",\
    'C-USA':"M",\
    'Single _United_States_of_America_Korea':"S",\
    'Common_Law_USA':"U",\
    'Domestic_Partner_United_States_of_America':"M",\
    'M-GBR':"M",\
    'Married_Canada':"M",\
    'Single_United_Kingdom':"S",\
    "USA_Separated" : "S",\
    'USA_Partnered': "M",\
    'M-DEU': "M",\
    'MARITAL_STATUS-6-300': "U",\
    'Divorced_United_States_of_America_Japan': "D",\
    'Married_United_States_of_America_Germany': "M",\
    'Single_United_States_of_America_Germany': "S",\
    'MARITAL_STATUS-6-303': "U",\
    'Married _United_States_of_America_Korea': "M",\
    'CU': "U",\
    'MARITAL_STATUS-6-301': "U",\
    'Married_Spain': "M",\
    'Single_Spain': "S",\
    'Not_Indicated_United_States_of_America_Panama': "U",\
    'Single _United_States_of_America_United Kingdom': "S",\
    'Single_Argentina': "S",\
    'S-CAN': "S",\
    'MARITAL_STATUS-6-322': "U",\
    'M-MEX': "M",\
    'Civil_Partnership_USA': "M",\
    'USA-Separated' : "D",\
    'USA_Common-law' : "U",\
    'MARITAL_STATUS-3-321' : "U",\
    'Separated_USA' :"D",\
    'USA_Not Disclosed' : "U",\
    'Single_ITA':"S",\
    'Head_of_Household_United_States_of_America':"U",\
    'M-AUS':"M",\
    'USA-Widowed':"W",\
    'Married_India':"M",\
    'Married_Mexico':"M",\
    'Divorced_Mexico':"D",\
    'Single_Mexico':"S",\
    'Free_Union_Mexico':"M",\
    'Married_Puerto_Rico':"M",\
    'MARITAL_STATUS-6-115':"U",\
    'PR_Single':"S",\
    'Not Provided':"U",\
    'USA-Civil Partnership':"M",\
    'CAN_Single':"S",\
    'MARITAL_STATUS-6-321':"U",\
    'Unknown_Puerto Rico':"U",\
    'Domestic Partner_United_States_of_America':"M",\
    'Surviving_Civil_Partner_United_States_of_America':"M",\
    np.nan:"U"
    }
    check_for_not_found(column, mapper, "marital_status")
    return column.map(mapper)

def clean_mapped_fullpart_description(column):
    mapper = {
        "Full Time" : "Full Time",\
        "Part Time" : "Part Time",\
        "Full-Time" : "Full Time",\
        "Part-Time" : "Part Time",\
        np.nan:"Unknown"
    }
    check_for_not_found(column, mapper, "mapped_fullpart_description")
    return column.map(mapper)

def clean_mapped_fullpart_code(column):
    mapper = {
        "FLTM": "FT",\
        "PRTM": "PR",\
        "FT": "FT",\
        "PT": "PT",\
        "DNM": "D",\
        np.nan:"Unknown"
    }
    check_for_not_found(column, mapper, "mapped_fullpart_code")
    return column.map(mapper)

def clean_mapped_permanent_temporary_code(column):
    '''
    Fill in the X
    '''
    mapper = {
        "PERM" : "P",\
        "R" : "R",\
        "TEMP" : "T",\
        "T" : "T",\
        "FR" : "FR",\
        "DNM" : "X",\
        "FT" : "FT",\
        np.nan:"Unknown"
    }
    check_for_not_found(column, mapper, "mapped_permanent_temporary_code")
    return column.map(mapper)


def clean_base_pay_regular_frequency_code(column):
    mapper = {
        'A1':"1" ,\
        'H1':"2" ,\
        '3':"3" ,\
        '4':"4" ,\
        'Annual':"5" ,\
        'M2':"6" ,\
        '2':"7" ,\
        'W1':"8" ,\
        'M1':" 9",\
        '1':"10" ,\
        'W2':"11" ,\
        'Unknown' :"0",\
        np.nan :"0" 
    }
    check_for_not_found(column, mapper, "base_pay_regular_frequency_code")
    return column.map(mapper)

def clean_base_pay_regular_expectedannualsalary_range(column):
    mapper = {
        '<$20,000': "1",\
        '$80,000-$99,999' : "2",\
        '$20,000-$39,999' : "3",\
        '$40,000-$59,999' : "4",\
        '$100,000-$174,999': "5",\
        '$60,000-$79,999' : "6",\
        '$175,000-$249,999' : "7",\
        '$250,000 +' : "8",\
        'Unknown' : "0",\
        np.nan : "0" 
    }
    check_for_not_found(column, mapper, "base_pay_regular_expectedannualsalary_range")
    return column.map(mapper)

def clean_platform_indicator_code(column):
    mapper = {
        'R4': "1",\
        'HM': "2",\
        'DCE': "3",\
        'R3': "4",\
        'HWE4': "5",\
        'DBE': "6",\
        np.nan : "0" 
    }
    check_for_not_found(column, mapper, "platform_indicator_code")
    return column.map(mapper)

def clean_mapped_flex_status_code(column):
    mapper = {
        "ELIGIBLE":"1",\
        "DNM":"2",\
        "NOTELIGIBLE":"3",\
        "ACTF":"4",\
        "INELIG":"3",\
        "HIITSD" : "5",\
        np.nan :"0" 
    }
    check_for_not_found(column, mapper, "mapped_flex_status_code")
    return column.map(mapper)

def clean_mapped_employment_status_code(column):
    mapper = {
        'ACTIVE': "1",\
        'TERM': "2",\
        'LTD': "3",\
        'RTEE': "4",\
        'Terminated': "2",\
        'Active': "1",\
        'DCSD_INSV': "7",\
        'LOA_WITH_PAY': "8",\
        'SVRN_PAY': "9",\
        'LOA_NO_PAY': "10",\
        'LOA - Unpaid Leave': "11",\
        'STD': "12",\
        'LOFF': "13",\
        'Retired': "14",\
        'COBRA_EE': "15",\
        'DCSD_OUT_OF_SV': "16",\
        'LOA - Paid Leave': "17",\
        'LOA_FM': "18",\
        'LOA_MLTR': "19",\
        'TERMINATED': "2",\
        'ENRL_QDRO': "21",\
        'COBRA_DPND': "22",\
        'ENRL_BENE': "23",\
        'LOA - STD': "24",\
        'LOA - Unpaid w/ Benefits': "25",\
        'Terminated w/ Benefits': "26",\
        'LOA - LTD': "27",\
        'LOA - Workers Comp': "28",\
        'DNM': "29",\
        'LOA_WC': "28",\
        'LOA - UNPAID W/ BENEFITS': "25",\
        'RETIRED': "14",\
        'Temporary': "33",\
        'LOANP': "10",\
        "Surviving Spouse": "35",\
        'Surviving Dependent':"35",\
        'Suspended':"36",\
        np.nan:"0"
    }
    check_for_not_found(column, mapper, "mapped_employment_status_code")
    return column.map(mapper)


def clean_is_rehire(column):
    mapper = {
        "Y" : "1",\
        "N" : "2",\
        np.nan:"0"
    }
    check_for_not_found(column, mapper, "is_rehire")
    return column.map(mapper)

def clean_employment_status(column):
    mapper = {
        "Active" : "1",\
        "Inactive" : "2",\
        "Non-Employee" : "3",\
        "Unknown" : "4",\
        np.nan:"4"
    }
    check_for_not_found(column, mapper, "employment_status")
    return column.map(mapper)

def clean_mapped_hourly_salary_code(column):
    mapper = {
        "HRLY" : "H",\
        "SLRY" : "S",\
        "H" : "H",\
        "S" : "S",\
        "DNM" : "X",\
        np.nan:"Unknown"
    }
    check_for_not_found(column, mapper, "mapped_hourly_salary_code")
    return column.map(mapper)


def clean_plan_status(column):
    '''
    
    '''
    mapper = {
        "ACTIVE" : "1",\
        "ELIG" : "0",\
#         np.nan:"Unknown"
    }
    check_for_not_found(column, mapper, "plan_status")
    return column.map(mapper)

def clean_is_union(column):

    mapper = {
        "Y" : "1",\
        "N" : "0",\
#         np.nan:"Unknown"
    }
    check_for_not_found(column, mapper, "is_union")
    return column.map(mapper)

def clean_match_threshhold_flag(column):
    '''
    
    '''
    mapper = {
        "Above" : "1",\
        "Below" : "2",\
        "Equal" : "3",\
        "Unknown":"0",\
        np.nan:"0"
    }
    check_for_not_found(column, mapper, "match_threshhold_flag")
    return column.map(mapper)

def clean_participation_status(column):
    '''
    The target variable which needs to be predicted
    '''
    mapper = {
        "Active - Contributing" : "1",\
        "Eligible - Not Contributing" : "0",\
        "Active - Not Contributing" : "0",\
        "Eligible - Not Contributing w/ Balance" : "0",\
        "Eligible - Contributing" : "1",\
#         np.nan:"Unknown"
    }
    check_for_not_found(column, mapper, "participation_status")
    return column.map(mapper)


def clean_mapped_hourly_salary_description(column):
    mapper = {
        "Hourly" : "H",\
        "Salaried" : "S",\
        "Salary" : "S",\
        np.nan:"Unknown"
    }
    check_for_not_found(column, mapper, "mapped_hourly_salary_description")
    return column.map(mapper)

def clean_country_description(column):
    mapper = {
    "United States of America":"United States of America",\
    "United States":"United States of America",\
    np.nan:"Unknown",\
    "UNITED STATES":"United States of America",\
    "Mexico":"Mexico",\
    "Puerto Rico":"",\
    "Austria":"Austria",\
    "GERMANY":"Germany",\
    "Kuwait (State of Kuwait)":"Kuwait",\
    "Switzerland (Swiss Confederation)":"Switzerland",\
    "Germany (Federal Republic of Germany)":"Germany",\
    "India (Republic of India)":"India",\
    "United Kingdom (of Great Britain & Nrthn Ireland)":"United Kingdom",\
    "Canada":"Canada",\
    "Mexico (United Mexican States)":"Mexico",\
    "Saudi Arabia (Kingdom of Saudi Arabia)":"Saudi Arabia",\
    "Romania":"Romania",\
    "Brazil":"Brazil",\
    "Hong Kong Special Administrative Region of China":"Hong Kong",\
    "Argentina":"Argentina",\
    "Russia Federation":"Russia",\
    "Japan":"Japan",\
    "Angola":"Angola",\
    "Netherlands (Kingdom of the Netherlands)":"Netherlands",\
    "Ireland":"Ireland",\
    "Croatia":"Croatia",\
    "Republic of Serbia":"Serbia",\
    "Malaysia":"Malaysia",\
    "Thailand (Kingdom of Thailand)":"Thailand",\
    "Luxembourg (Grand Duchy of Luxembourg)":"Luxembourg",\
    "Gabon (Gabonese Republic)":"Gabon",\
    "France (French Republic)":"France",\
    "Spain (Kingdom of Spain)":"Spain",\
    "Poland (Republic of Poland)":"Poland",\
    "Egypt (Arab Republic of Egypt)":"Egypt",\
    "Singapore":"Singapore",\
    "Singapore (Republic of Singapore)":"Singapore",\
    "United Kingdom":"United Kingdom",\
    "Norway (Kingdom of Norway)":"Norway",\
    "Italy (Italian Republic)":"Italy",\
    "Palau (Republic of Palau)":"Palau",\
    "Qatar (State of Qatar)":"Qatar",\
    "Belgium":"Belgium",\
    "TX":"United States of America",\
    "Australia":"Australia",\
    "KOREA, REPUBLIC OF":"South Korea",\
    "Lithuania (Republic of Lithuania)":"Lithuania",\
    "Korea (Republic Of)":"South Korea",\
    "Colombia":"Colombia",\
    "Turkey (Republic of Turkey)":"Turkey",\
    "Israel (State of Israel)":"Israel",\
    "Hong Kong":"Hong Kong",\
    "V9pjc/xEqEDBWKPIajwbNQ==":"Unknown",\
    "China":"China",\
    "Uruguay":"Uruguay",\
    "Trinidad and Tobago (Republic of)":"Trinidad and Tobago",\
    "JAPAN":"Japan",\
    "Iceland (Republic of Iceland)":"Iceland",\
    "United Arab Emirates":"United Arab Emirates",\
    "Sweden (Kingdom of Sweden)":"Sweden",\
    "India":"India",\
    "Indonesia (Republic of Indonesia)":"Indonesia",\
    "Iraq (Republic of Iraq)":"Iraq",\
    "Czech Republic":"Czech Republic",\
    "UNITED KINGDOM":"United Kingdom",\
    "Bermuda":"Bermuda",\
    "Panama (Republic of Panama)":"Panama",\
    "Algeria":"Algeria",\
    "Spain":"Spain",\
    "Bangladesh":"Bangladesh",\
    "Philippines (Republic of Philippines)":"Philippines",\
    "Jamaica":"Jamaica",\
    "New Zealand":"New Zealand",\
    "Costa Rica":"Costa Rica",\
    "Republic of Guyana":"Guyana",\
    "Jordan (Hashemite Kingdom of Jordan)":"Jordan",\
    "Cook Islands":"Cook Islands",\
    "Ecuador":"Ecuador",\
    "Germany":"Germany",\
    "Republic of the Gambia":"Gambia",\
    "Denmark":"Denmark",\
    "Italy":"Italy",\
    "Hungary (Republic of Hungary)":"Hungary",\
    "American Samoa":"American Samoa",\
    "Virgin Islands (U.S.)":"United States of America",\
    "Vietnam (Socialist Republic of Viet Nam)":"Vietnam",\
    "Bolivarian Republic of Venezuela":"Venezuela",\
    "Portugal (Portuguese Republic)":"Portugal",\
        'POLAND' : "Poland",\
        'Piscataway':"United States of America",\
        'Taiwan':"Taiwan",\
        "":"Unknown",\
        np.nan:"Unknown"
    }
    check_for_not_found(column, mapper, "country_description")
    return column.map(mapper)

def clean_state(column):
    mapper = {
    "AL":"AL",\
    "AK":"AK",\
    "AR":"AR",\
    "AZ":"AZ",\
    "CA":"CA",\
    "CO":"CO",\
    "CT":"CT",\
    "DE":"DE",\
    "DC":"DC",\
    "FL":"FL",\
    "GA":"GA",\
    "HI":"HI",\
    "ID":"ID",\
    "IL":"IL",\
    "IN":"IN",\
    "IA":"IA",\
    "KS":"KS",\
    "KY":"KY",\
    "LA":"LA",\
    "ME":"ME",\
    "MD":"MD",\
    "MA":"MA",\
    "MI":"MI",\
    "MN":"MN",\
    "MS":"MS",\
    "MO":"MO",\
    "MT":"MT",\
    "NE":"NE",\
    "NV":"NV",\
    "NH":"NH",\
    "NJ":"NJ",\
    "NM":"NM",\
    "NY":"NY",\
    "NC":"NC",\
    "ND":"ND",\
    "OH":"OH",\
    "OK":"OK",\
    "OR":"OR",\
    "PA":"PA",\
    "RI":"RI",\
    "SC":"SC",\
    "SD":"SD",\
    "TN":"TN",\
    "TX":"TX",\
    "UT":"UT",\
    "VT":"VT",\
    "VA":"VA",\
    "WA":"WA",\
    "WV":"WV",\
    "WI":"WI",\
    "WY":"WY",\
    "al":"AL",\
    "ak":"AK",\
    "ar":"AR",\
    "az":"AZ",\
    "ca":"CA",\
    "co":"CO",\
    "ct":"CT",\
    "de":"DE",\
    "dc":"DC",\
    "fl":"FL",\
    "ga":"GA",\
    "hi":"HI",\
    "id":"ID",\
    "il":"IL",\
    "in":"IN",\
    "ia":"IA",\
    "ks":"KS",\
    "ky":"KY",\
    "la":"LA",\
    "me":"ME",\
    "md":"MD",\
    "ma":"MA",\
    "mi":"MI",\
    "mn":"MN",\
    "ms":"MS",\
    "mo":"MO",\
    "mt":"MT",\
    "ne":"NE",\
    "nv":"NV",\
    "nh":"NH",\
    "nj":"NJ",\
    "nm":"NM",\
    "ny":"NY",\
    "nc":"NC",\
    "nd":"ND",\
    "oh":"OH",\
    "ok":"OK",\
    "or":"OR",\
    "pa":"PA",\
    "ri":"RI",\
    "sc":"SC",\
    "sd":"SD",\
    "tn":"TN",\
    "tx":"TX",\
    "ut":"UT",\
    "vt":"VT",\
    "va":"VA",\
    "wa":"WA",\
    "wv":"WV",\
    "wi":"WI",\
    "wy":"WY",\
        np.nan:"Unknown"
    }
    check_for_not_found(column, mapper, "state")
    return column.map(mapper)

