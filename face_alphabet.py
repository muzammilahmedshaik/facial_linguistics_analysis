#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and datafiles

# In[2]:


import pandas as pd
import numpy as np
x =  pd.read_csv("C:\\Users\\hp\\Downloads\\micro_high_res_every_frame.csv")
y = pd.read_csv("C:\\Users\\hp\\Downloads\\updated_fer.csv")


# # Conv Fer to range from 1- 100

# In[3]:



y[["angry","disgust","fear","happy","sad","surprise","neutral"]] *= 100 


# # 1) Compute model averaged sentiment scores for each individual sentiment type from different models (FER, DeepFace etc)

# In[4]:


# fer_mean = y.mean(axis=0)
deep_mean = x[["df_angry","df_disgust","df_fear","df_happy","df_sad","df_surprise","df_neutral"]].mean(axis=0)
# fer_mean = pd.DataFrame(fer_mean)
deep_mean = pd.DataFrame(deep_mean)


# In[5]:


# fer_mean=fer_mean.iloc[1:]
# fer_mean=fer_mean.T
# print(fer_mean)


# In[6]:


deep_mean = deep_mean.T
print(deep_mean)


# # Compute Storyline negativity scores for StoryTIME and screenPLAY.

# In[7]:


DF_Overall_Negativitys_core = deep_mean[["df_angry","df_disgust","df_fear","df_sad"]].mean(axis=1)
DF_Overall_Negativitys_core = pd.DataFrame(DF_Overall_Negativitys_core)
DF_Overall_Negativitys_core.columns = ["DF_Overall_Negativitys_core"]
DF_Overall_Negativitys_core.head()


# In[8]:


DF_Overall_Aggression_score = deep_mean[["df_angry","df_disgust"]].mean(axis=1)
DF_Overall_Aggression_score = pd.DataFrame(DF_Overall_Aggression_score)
DF_Overall_Aggression_score.columns = ["DF_Overall_Aggression_score"]
print(DF_Overall_Aggression_score)


# In[9]:


DF_Depression_score = deep_mean[["df_sad","df_fear"]].mean(axis=1)
DF_Depression_score = pd.DataFrame(DF_Depression_score)
DF_Depression_score.columns = ["DF_Depression_score"]
print(DF_Depression_score)


# # Compute Storyline.face.sentiment score for each frame (row) by dividing the happiness score by the sum of the happiness + negative sentiment scores. This score shows the percentage of net face sentiment that is positive (0% to 100%). Put into StoryTIME.
# 

# In[10]:


Storyline_face_sentiment = (x[["df_happy"]] /(x[["df_happy"]] + DF_Overall_Negativitys_core["DF_Overall_Negativitys_core"][0]))*100
Storyline_face_sentiment.columns = ["Storyline_face_sentiment"]
Storyline_face_sentiment.head()


# # Add Core Facial expression scores to storyTIME and screenPLAY
# 

# In[11]:


Core_Expression_Scores_DeepFace_Model_y = pd.concat([deep_mean,DF_Overall_Negativitys_core,DF_Overall_Aggression_score,DF_Depression_score,Storyline_face_sentiment],axis=1)
Core_Expression_Scores_DeepFace_Model_y.head()


# # Save DataFrame in CSV

# In[12]:


# Core_Expression_Scores_DeepFace_Model_y.to_csv (r"D:\Master's\Intern\Core_Expression_Scores_DeepFace_Model_y.csv", index = False, header=True)


# # ******* Ignore this part as it was used for earlier version of the doc********

# In[13]:


# avg_modal = pd.DataFrame(np.vstack([fer_mean, deep_mean]), columns = list(fer_mean.columns))
# avg_modal.head()


# In[14]:


# avg_modal = avg_modal.mean(axis=0)
# print(avg_modal)


# # 2) Define positivity score for each frame (row) in each video at 24 fps from the average of smiling and happiness scores from different models. (rows)

# In[15]:


# fer = y[['angry',   'disgust',      'fear',     'happy',       'sad',  'surprise',   'neutral']]
# deep =  x[["df_angry","df_disgust","df_fear","df_happy","df_sad","df_surprise","df_neutral"]]
# fer.head()


# In[16]:


# deep.head()


# In[17]:


# deep = deep.rename({'df_happy': 'happy','df_angry':'angry','df_disgust':'disgust','df_fear':'fear','df_sad':'sad','df_surprise':'surprise','df_neutral':'neutral'}, axis=1)
# deep.head()


# In[18]:


# positive_score=pd.concat([deep[["happy"]],fer[["happy"]]],axis=1)
# positive_score.head()


# In[19]:


# positive_score['positive_score'] = positive_score.mean(axis=1)
# positive_score.head()


# # 3)Compute negativity score from the average of sadness, anger, disgust, contempt and fear scores from different models for each frame (row) in each video at 24 fps. (row)

# In[20]:


# negative_score=pd.concat([deep[["angry","disgust","sad","fear"]],fer[["angry","disgust","sad","fear"]]],axis=1)
# negative_score['negative_score'] = negative_score.mean(axis=1)
# negative_score.head()


# # 4)Compute face.sentiment score for each frame (row) by dividing the positivity score by the sum of the positive + negative sentiment scores. This score shows the percentage of net face sentiment that is positive (0% to 100%).

# In[21]:


# face_sentiment_score = positive_score['positive_score'] / negative_score['negative_score'] + positive_score['positive_score']


# In[22]:


# face_sentiment_score["face_sentiment_score"]= pd.DataFrame(face_sentiment_score)



# # ******* Ignore till here as it was used for earlier version of the doc********

# # Face Alphabet - Facial Sentiment Analysis Algorithm

# # Compute the question (Q) level facial sentiment score from all frames for a given question

# In[23]:


facial_sentimental_report = x[["subject_id","assessment_id","question_id","question_type","main_question","start_answer_datetime","end_answer_datetime","df_angry","df_disgust","df_fear","df_happy","df_sad","df_surprise","df_neutral"]]
facial_sentimental_report.head()


# In[24]:


facial_sentimental_report_Q = facial_sentimental_report.groupby(['question_id']).sum()
Q_Face_Sentiment_Score_QuestionID = facial_sentimental_report_Q[["df_happy"]]/(facial_sentimental_report_Q[["df_happy"]]+DF_Overall_Negativitys_core["DF_Overall_Negativitys_core"][0])

Q_Face_Sentiment_Score_QuestionID.columns = ["Q_Face_Sentiment_Score_QuestionID"]
print(Q_Face_Sentiment_Score_QuestionID)


# # Compute the assessment (A) level facial sentiment score from all frames for a given assessment
# 

# In[25]:


facial_sentimental_report_A = facial_sentimental_report.groupby(['assessment_id']).sum()
A_Face_Sentiment_Score_AssessmentID = facial_sentimental_report_A[["df_happy"]]/(facial_sentimental_report_A[["df_happy"]]+DF_Overall_Negativitys_core["DF_Overall_Negativitys_core"][0])
A_Face_Sentiment_Score_AssessmentID.columns = ["A_Face_Sentiment_Score_AssessmentID"]
print(A_Face_Sentiment_Score_AssessmentID)


# # Compute the total emotion score by summing the scores for each micro-expression for every frame by question and assessment.

# In[26]:


total_Facial_Emotion_Score_questionID = facial_sentimental_report.groupby(['question_id']).sum()
total_Facial_Emotion_Score_questionID_x = total_Facial_Emotion_Score_questionID.sum(axis=1)

total_Facial_Emotion_Score_questionID_x = pd.DataFrame(total_Facial_Emotion_Score_questionID_x).reset_index()
total_Facial_Emotion_Score_questionID_x = total_Facial_Emotion_Score_questionID_x.rename(columns= {0: 'total_Facial_Emotion_Score_questionID'})
total_Facial_Emotion_Score_questionID_x = total_Facial_Emotion_Score_questionID_x.assign(Index=range(len(total_Facial_Emotion_Score_questionID_x))).set_index('Index')

total_Facial_Emotion_Score_questionID_x


# In[27]:


total_Facial_Emotion_Score_assessmentID = facial_sentimental_report.groupby(['assessment_id']).sum()
total_Facial_Emotion_Score_assessmentID_x = total_Facial_Emotion_Score_assessmentID.sum(axis=1)
total_Facial_Emotion_Score_assessmentID_x = pd.DataFrame(total_Facial_Emotion_Score_assessmentID_x).reset_index()
total_Facial_Emotion_Score_assessmentID_x = total_Facial_Emotion_Score_assessmentID_x.rename(columns= {0: 'total_Facial_Emotion_Score_assessmentID'})
total_Facial_Emotion_Score_assessmentID_x = total_Facial_Emotion_Score_assessmentID_x.assign(Index=range(len(total_Facial_Emotion_Score_assessmentID_x))).set_index('Index')

total_Facial_Emotion_Score_assessmentID_x


# # Compute the following Storyline Net Emotion scores for each question by summing the scores for each micro-expression across all frames by question
# 

# In[28]:


Storyline_Net_Emotion_scores = facial_sentimental_report.groupby(['question_id']).sum()

Storyline_angry_Sum_questionID = Storyline_Net_Emotion_scores[['df_angry']].reset_index(drop=True)
Storyline_angry_Sum_questionID.columns = ["Storyline_angry_Sum_questionID"]
Storyline_angry_Sum_questionID = Storyline_angry_Sum_questionID.assign(Index=range(len(Storyline_angry_Sum_questionID))).set_index('Index')

Storyline_disgust_Sum_questionID = Storyline_Net_Emotion_scores[['df_disgust']].reset_index(drop=True)
Storyline_disgust_Sum_questionID.columns = ["Storyline_disgust_Sum_questionID"]
Storyline_disgust_Sum_questionID = Storyline_disgust_Sum_questionID.assign(Index=range(len(Storyline_disgust_Sum_questionID))).set_index('Index')

Storyline_fear_Sum_questionID = Storyline_Net_Emotion_scores[['df_fear']].reset_index(drop=True)
Storyline_fear_Sum_questionID.columns = ["Storyline_fear_Sum_questionID"]
Storyline_fear_Sum_questionID = Storyline_fear_Sum_questionID.assign(Index=range(len(Storyline_fear_Sum_questionID))).set_index('Index')

Storyline_happy_Sum_questionID = Storyline_Net_Emotion_scores[['df_happy']].reset_index(drop=True)
Storyline_happy_Sum_questionID.columns = ["Storyline_happy_Sum_questionID"]
Storyline_happy_Sum_questionID = Storyline_happy_Sum_questionID.assign(Index=range(len(Storyline_happy_Sum_questionID))).set_index('Index')

Storyline_sad_Sum_questionID = Storyline_Net_Emotion_scores[['df_sad']].reset_index(drop=True)
Storyline_sad_Sum_questionID.columns = ["Storyline_sad_Sum_questionID"]
Storyline_sad_Sum_questionID = Storyline_sad_Sum_questionID.assign(Index=range(len(Storyline_sad_Sum_questionID))).set_index('Index')

Storyline_surprise_Sum_questionID = Storyline_Net_Emotion_scores[['df_surprise']].reset_index(drop=True)
Storyline_surprise_Sum_questionID.columns = ["Storyline_surprise_Sum_questionID"]
Storyline_surprise_Sum_questionID = Storyline_surprise_Sum_questionID.assign(Index=range(len(Storyline_surprise_Sum_questionID))).set_index('Index')
Storyline_Neutral_Sum_questionID = Storyline_Net_Emotion_scores[['df_neutral']].reset_index(drop=True)
Storyline_Neutral_Sum_questionID.columns = ["Storyline_Neutral_Sum_questionID"]
Storyline_Neutral_Sum_questionID = Storyline_Neutral_Sum_questionID.assign(Index=range(len(Storyline_Neutral_Sum_questionID))).set_index('Index')

Storyline_Neutral_Sum_questionID


# # Compute the following Storyline Net Emotion scores for each assessment by summing the scores for each micro-expression across all frames in an assessment

# In[29]:


Storyline_Net_Emotion_scores_a = facial_sentimental_report.groupby(['assessment_id']).sum()

Storyline_angry_Sum_assessmentID = Storyline_Net_Emotion_scores_a[['df_angry']].reset_index(drop=True)
Storyline_angry_Sum_assessmentID.columns = ["Storyline_angry_Sum_assessmentID"]
Storyline_angry_Sum_assessmentID = Storyline_angry_Sum_assessmentID.assign(Index=range(len(Storyline_angry_Sum_assessmentID))).set_index('Index')

Storyline_disgust_Sum_assessmentID = Storyline_Net_Emotion_scores_a[['df_disgust']].reset_index(drop=True)
Storyline_disgust_Sum_assessmentID.columns = ["Storyline_disgust_Sum_assessmentID"]
Storyline_disgust_Sum_assessmentID = Storyline_disgust_Sum_assessmentID.assign(Index=range(len(Storyline_disgust_Sum_assessmentID))).set_index('Index')

Storyline_fear_Sum_assessmentID = Storyline_Net_Emotion_scores_a[['df_fear']].reset_index(drop=True)
Storyline_fear_Sum_assessmentID.columns = ["Storyline_fear_Sum_assessmentID"]
Storyline_fear_Sum_assessmentID = Storyline_fear_Sum_assessmentID.assign(Index=range(len(Storyline_fear_Sum_assessmentID))).set_index('Index')

Storyline_happy_Sum_assessmentID = Storyline_Net_Emotion_scores_a[['df_happy']].reset_index(drop=True)
Storyline_happy_Sum_assessmentID.columns = ["Storyline_happy_Sum_assessmentID"]
Storyline_happy_Sum_assessmentID = Storyline_happy_Sum_assessmentID.assign(Index=range(len(Storyline_happy_Sum_assessmentID))).set_index('Index')

Storyline_sad_Sum_assessmentID = Storyline_Net_Emotion_scores_a[['df_sad']].reset_index(drop=True)
Storyline_sad_Sum_assessmentID.columns = ["Storyline_sad_Sum_assessmentID"]
Storyline_sad_Sum_assessmentID = Storyline_sad_Sum_assessmentID.assign(Index=range(len(Storyline_sad_Sum_assessmentID))).set_index('Index')

Storyline_surprise_Sum_assessmentID = Storyline_Net_Emotion_scores_a[['df_surprise']].reset_index(drop=True)
Storyline_surprise_Sum_assessmentID.columns = ["Storyline_surprise_Sum_assessmentID"]
Storyline_surprise_Sum_assessmentID = Storyline_surprise_Sum_assessmentID.assign(Index=range(len(Storyline_surprise_Sum_assessmentID))).set_index('Index')

Storyline_Neutral_Sum_assessmentID = Storyline_Net_Emotion_scores_a[['df_neutral']].reset_index(drop=True)
Storyline_Neutral_Sum_assessmentID.columns = ["Storyline_Neutral_Sum_assessmentID"]
Storyline_Neutral_Sum_assessmentID = Storyline_Neutral_Sum_assessmentID.assign(Index=range(len(Storyline_Neutral_Sum_assessmentID))).set_index('Index')
Storyline_Neutral_Sum_assessmentID


# # Compute the following proportional facial emotion scores for each question by summing the scores for each micro-expression and normalizing to the Total Emotion Score (above)
# 

# In[74]:


Proportion_Neutral_questionID = Storyline_Neutral_Sum_questionID['Storyline_Neutral_Sum_questionID'] /  total_Facial_Emotion_Score_questionID_x['total_Facial_Emotion_Score_questionID']
Proportion_Neutral_questionID = pd.DataFrame(Proportion_Neutral_questionID)
Proportion_Neutral_questionID.columns = ["Proportion_Neutral_questionID"]
Proportion_Neutral_questionID


# In[76]:


Proportion_Surprise_questionID = Storyline_surprise_Sum_questionID['Storyline_surprise_Sum_questionID'] / total_Facial_Emotion_Score_questionID_x['total_Facial_Emotion_Score_questionID']
Proportion_Surprise_questionID = pd.DataFrame(Proportion_Surprise_questionID)
Proportion_Surprise_questionID.columns = ["Proportion_Surprise_questionID"]
Proportion_Surprise_questionID


# In[77]:


Proportion_Sad_questionID = Storyline_sad_Sum_questionID['Storyline_sad_Sum_questionID'] / total_Facial_Emotion_Score_questionID_x['total_Facial_Emotion_Score_questionID']
Proportion_Sad_questionID = pd.DataFrame(Proportion_Sad_questionID)
Proportion_Sad_questionID.columns = ["Proportion_Sad_questionID"]
Proportion_Sad_questionID


# In[78]:


Proportion_Happy_questionID = Storyline_happy_Sum_questionID['Storyline_happy_Sum_questionID'] / total_Facial_Emotion_Score_questionID_x['total_Facial_Emotion_Score_questionID']
Proportion_Happy_questionID = pd.DataFrame(Proportion_Happy_questionID)
Proportion_Happy_questionID.columns = ["Proportion_Happy_questionID"]
Proportion_Happy_questionID


# In[79]:


Proportion_Fear_questionID = Storyline_fear_Sum_questionID['Storyline_fear_Sum_questionID'] / total_Facial_Emotion_Score_questionID_x['total_Facial_Emotion_Score_questionID']
Proportion_Fear_questionID = pd.DataFrame(Proportion_Fear_questionID)
Proportion_Fear_questionID.columns = ["Proportion_Fear_questionID"]
Proportion_Fear_questionID


# In[80]:


Proportion_Angry_questionID = Storyline_angry_Sum_questionID['Storyline_angry_Sum_questionID'] / total_Facial_Emotion_Score_questionID_x['total_Facial_Emotion_Score_questionID']
Proportion_Angry_questionID = pd.DataFrame(Proportion_Angry_questionID)
Proportion_Angry_questionID.columns = ["Proportion_Angry_questionID"]
Proportion_Angry_questionID


# In[81]:


Proportion_Disgust_questionID = Storyline_disgust_Sum_questionID['Storyline_disgust_Sum_questionID'] / total_Facial_Emotion_Score_questionID_x['total_Facial_Emotion_Score_questionID']
Proportion_Disgust_questionID = pd.DataFrame(Proportion_Disgust_questionID)
Proportion_Disgust_questionID.columns = ["Proportion_Disgust_questionID"]
Proportion_Disgust_questionID


# # Compute the following proportional facial emotion scores for each assessment by summing the scores for each micro-expression and normalizing to the Total Emotion Score (above)
# 

# In[96]:


Proportion_Happy_assessmentID = Storyline_happy_Sum_assessmentID['Storyline_happy_Sum_assessmentID'] / total_Facial_Emotion_Score_assessmentID_x['total_Facial_Emotion_Score_assessmentID']
Proportion_Happy_assessmentID = pd.DataFrame(Proportion_Happy_assessmentID)
Proportion_Happy_assessmentID.columns = ["Proportion_Happy_assessmentID"]
Proportion_Happy_assessmentID


# In[97]:


Proportion_Angry_assessmentID = Storyline_angry_Sum_assessmentID['Storyline_angry_Sum_assessmentID'] / total_Facial_Emotion_Score_assessmentID_x['total_Facial_Emotion_Score_assessmentID']
Proportion_Angry_assessmentID = pd.DataFrame(Proportion_Angry_assessmentID)
Proportion_Angry_assessmentID.columns = ["Proportion_Angry_assessmentID"]
Proportion_Angry_assessmentID


# In[98]:


Proportion_Disgust_assessmentID = Storyline_disgust_Sum_assessmentID['Storyline_disgust_Sum_assessmentID'] / total_Facial_Emotion_Score_assessmentID_x['total_Facial_Emotion_Score_assessmentID']
Proportion_Disgust_assessmentID = pd.DataFrame(Proportion_Disgust_assessmentID)
Proportion_Disgust_assessmentID.columns = ["Proportion_Disgust_assessmentID"]
Proportion_Disgust_assessmentID


# In[99]:


Proportion_Fear_assessmentID = Storyline_fear_Sum_assessmentID['Storyline_fear_Sum_assessmentID'] / total_Facial_Emotion_Score_assessmentID_x['total_Facial_Emotion_Score_assessmentID']
Proportion_Fear_assessmentID = pd.DataFrame(Proportion_Fear_assessmentID)
Proportion_Fear_assessmentID.columns = ["Proportion_Fear_assessmentID"]
Proportion_Fear_assessmentID


# In[100]:


Proportion_Sad_assessmentID = Storyline_sad_Sum_assessmentID['Storyline_sad_Sum_assessmentID'] / total_Facial_Emotion_Score_assessmentID_x['total_Facial_Emotion_Score_assessmentID']
Proportion_Sad_assessmentID = pd.DataFrame(Proportion_Sad_assessmentID)
Proportion_Sad_assessmentID.columns = ["Proportion_Sad_assessmentID"]
Proportion_Sad_assessmentID


# In[83]:


Proportion_Surprise_assessmentID = Storyline_surprise_Sum_assessmentID['Storyline_surprise_Sum_assessmentID'] / total_Facial_Emotion_Score_assessmentID_x['total_Facial_Emotion_Score_assessmentID']
Proportion_Surprise_assessmentID = pd.DataFrame(Proportion_Surprise_assessmentID)
Proportion_Surprise_assessmentID.columns = ["Proportion_Surprise_assessmentID"]
Proportion_Surprise_assessmentID


# In[82]:


Proportion_Neutral_assessmentID = Storyline_Neutral_Sum_assessmentID['Storyline_Neutral_Sum_assessmentID'] / total_Facial_Emotion_Score_assessmentID_x['total_Facial_Emotion_Score_assessmentID']
Proportion_Neutral_assessmentID = pd.DataFrame(Proportion_Neutral_assessmentID)
Proportion_Neutral_assessmentID.columns = ["Proportion_Neutral_assessmentID"]
Proportion_Neutral_assessmentID


# # Creating two csv files for assessments and questions

# In[58]:


subject_id = facial_sentimental_report.groupby(['subject_id']).sum()
subject_id =pd.DataFrame(subject_id.index)
assessment_id = facial_sentimental_report.groupby(['assessment_id']).sum()
assessment_id =pd.DataFrame(assessment_id.index)
question_id = facial_sentimental_report.groupby(['question_id']).sum()
question_id = pd.DataFrame(question_id.index)
question_type = facial_sentimental_report.groupby(['question_type']).sum()
question_type = pd.DataFrame(question_type.index)
main_question = facial_sentimental_report.groupby(['main_question']).sum()
main_question = pd.DataFrame(main_question.index)
question_id


# In[104]:



face_alpha_report = pd.merge(question_id, total_Facial_Emotion_Score_questionID_x, on="question_id")
face_alpha_report = pd.merge(face_alpha_report, Q_Face_Sentiment_Score_QuestionID, on="question_id")
face_alpha_report = face_alpha_report.join(Storyline_angry_Sum_questionID)
face_alpha_report = face_alpha_report.join(Storyline_disgust_Sum_questionID)
face_alpha_report = face_alpha_report.join(Storyline_fear_Sum_questionID)
face_alpha_report = face_alpha_report.join(Storyline_happy_Sum_questionID)
face_alpha_report = face_alpha_report.join(Storyline_sad_Sum_questionID)
face_alpha_report = face_alpha_report.join(Storyline_surprise_Sum_questionID)
face_alpha_report = face_alpha_report.join(Storyline_Neutral_Sum_questionID)
face_alpha_report = face_alpha_report.join(Proportion_Neutral_questionID)
face_alpha_report = face_alpha_report.join(Proportion_Surprise_questionID)
face_alpha_report = face_alpha_report.join(Proportion_Sad_questionID)
face_alpha_report = face_alpha_report.join(Proportion_Happy_questionID)
face_alpha_report = face_alpha_report.join(Proportion_Fear_questionID)
face_alpha_report = face_alpha_report.join(Proportion_Angry_questionID)
face_alpha_report_QID = face_alpha_report.join(Proportion_Disgust_questionID)
face_alpha_report_QID.to_csv (r"D:\Master's\Intern\face_alpha_report_QID.csv", index = False, header=True)


face_alpha_report_x = pd.merge(assessment_id, A_Face_Sentiment_Score_AssessmentID, on="assessment_id")
face_alpha_report_x = pd.merge(face_alpha_report_x, total_Facial_Emotion_Score_assessmentID_x, on="assessment_id")
face_alpha_report_x = face_alpha_report_x.join(Storyline_angry_Sum_assessmentID)
face_alpha_report_x = face_alpha_report_x.join(Storyline_disgust_Sum_assessmentID)
face_alpha_report_x = face_alpha_report_x.join(Storyline_fear_Sum_assessmentID)
face_alpha_report_x = face_alpha_report_x.join(Storyline_happy_Sum_assessmentID)
face_alpha_report_x = face_alpha_report_x.join(Storyline_sad_Sum_assessmentID)
face_alpha_report_x = face_alpha_report_x.join(Storyline_surprise_Sum_assessmentID)
face_alpha_report_x = face_alpha_report_x.join(Storyline_Neutral_Sum_assessmentID)
face_alpha_report_x = face_alpha_report_x.join(Proportion_Happy_assessmentID)
face_alpha_report_x = face_alpha_report_x.join(Proportion_Angry_assessmentID)
face_alpha_report_x = face_alpha_report_x.join(Proportion_Disgust_assessmentID)
face_alpha_report_x = face_alpha_report_x.join(Proportion_Fear_assessmentID)
face_alpha_report_x = face_alpha_report_x.join(Proportion_Sad_assessmentID)
face_alpha_report_x = face_alpha_report_x.join(Proportion_Surprise_assessmentID)
face_alpha_report_AID = face_alpha_report_x.join(Proportion_Neutral_assessmentID)
face_alpha_report_AID.to_csv (r"D:\Master's\Intern\face_alpha_report_AID.csv", index = False, header=True)


# In[93]:


face_alpha_report_QID


# In[102]:


face_alpha_report_AID


# In[ ]:




