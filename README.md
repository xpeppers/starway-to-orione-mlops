# Starway to Orione CUDA (Cloud Utilities and Data Analytics) - TEAM

DoAI is the **D**evelopment **O**peration and **A**rtificial **I**ntelligence

This is the learning path every new Cloud Data Architect has to follow when joining the XPeppers Cloud team.
This path reflects our team's culture and values, which have their roots in the [agile values and principles](http://agilemanifesto.org/).

* Flat Organizations:
   * [The Flat Way](https://link.medium.com/E4kjMXajO3) ```#onboarding```
   * [Teal Is The New Black: Self-management and the future of work](https://management30.com/blog/teal-organization-self-management-future-of-work/)
   * [Collaborative decision making in self-organizing teams](https://www.agilebusinessday.com/2019/09/26/collaborative-decision-making-in-self-organizing-teams-abd19-lorenzo-massacci/)
* Read chapters 1, 4, 5, 7 of [XP Explained](https://www.amazon.com/Extreme-Programming-Explained-Embrace-Change/dp/0201616416) ```#onboarding```
* Read chapters 2, 6 of [XP Explained](https://www.amazon.com/Extreme-Programming-Explained-Embrace-Change/dp/0201616416)
* Iterative and Incremental Development:
  * [Waterfall](https://condor.depaul.edu/dmumaugh/readings/handouts/IS375/IIDI.pdf) ```#onboarding```
  * [Agile](https://condor.depaul.edu/dmumaugh/readings/handouts/IS375/IIDII.pdf) ```#onboarding```
  * [Transition](https://condor.depaul.edu/dmumaugh/readings/handouts/IS375/IIDIII.pdf) 
* For __italian speakers__, Watch ["Perché è così difficile fare Extreme Programming"](https://vimeo.com/113090009) by Matteo Vaccari ```#onboarding```
* [Pair Programming](https://martinfowler.com/articles/on-pair-programming.html) ```#onboarding```
* Agile Mindset:
   * [What Exactly is the Agile Mindset?](https://www.infoq.com/articles/what-agile-mindset/) ```#onboarding```
* Read [The Pomodoro Technique](http://pomodorotechnique.com/) paper
* Read first chapter of ["Applying UML and Patterns"](http://www.amazon.com/Applying-UML-Patterns-Introduction-Object-Oriented/dp/0131489062)
  * Try to estimate the time needed to study that chapter (using the pomodoro technique)
  * Answer (for example on the team's wiki pages)
    * What is analysis?
    * What is design?
    * What's the difference between them?
    * What is design for?
      * in other words, how would you reply to the following statement: _"I just need to understand what to do (analysis) and then do it (coding). Everything else does not matter!"_

# Table of Contents

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#programming">Programming</a></li>
    <li><a href="#maths--statistics">Maths & Statistics</a></li>
    <li><a href="#toolboxes">Toolboxes</a></li>
    <li><a href="#machine-learning">Machine Learning</a></li>
    <li><a href="#deep-learning">Deep Learning</a></li>
    <li><a href="#cloud-operation-and-devops">Cloud Operation and DevOps</a></li> 
    <li><a href="#machine-learning-operations-mlops">Machine Learning Operations (MLOps)</a></li>
    <li><a href="#nlp--nlu">NLP & NLU</a></li>
    <li><a href="#recommendation-system">Recommendation System</a></li>
    <li><a href="#reinforcement-learning">Reinforcement Learning</a></li>
  </ol>
</details>

Please feel free to fork and contribute, add materials, fix the existing ones and propose new stuff.

During all the plan read [The Phoenix Project](https://books.google.it/books/about/The_Phoenix_Project.html?id=qaRODgAAQBAJ).

## Programming
* [Python course by Analytics Vidhya](https://courses.analyticsvidhya.com/courses/introduction-to-data-science)
* [Basic Python](https://www.learnpython.org/) - [Option 2: Basic Python](https://www.programiz.com/python-programming/)
* [Object-oriented Programming](https://realpython.com/python3-object-oriented-programming/)
* [Python Regular Expressions](https://developers.google.com/edu/python/regular-expressions)
* [RE Cheat Sheet](https://www.debuggex.com/cheatsheet/regex/python)
* [Git](https://learngitbranching.js.org/?locale=it_IT) - [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf):
    open source distributed version control system.
* [Shell Script](https://dagshub.com/blog/effective-linux-bash-data-scientists/)
* Competitive: [Starway to Orione
    Dev](https://github.com/xpeppers/starway-to-orione)

<a href="#table-of-contents">🠥🠥 Back to Table of Contents 🠥🠥</a>

## Maths & Statistics
* [Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
* [Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
* [Descriptive Statistics](https://conjointly.com/kb/descriptive-statistics/)
* [Data Distributions](https://towardsdatascience.com/7-statistical-distributions-that-every-data-scientist-should-know-with-intuitive-explanations-bf967db81f0b)
* [Convolutions](https://www.youtube.com/watch?v=8rrHTtUzyZA&list=PLZHQObOWTQDMp_VZelDYjka8tnXNpXhzJ&index=1)
* [Exploratory Data Analysis](https://medium.com/data-folks-indonesia/10-things-to-do-when-conducting-your-exploratory-data-analysis-eda-7e3b2dfbf812)
* [Regression](https://www.listendata.com/2018/03/regression-analysis.html)

<a href="#table-of-contents">🠥🠥 Back to Table of Contents 🠥🠥</a>

## Toolboxes
* Evaluating and Exploring data in Python
  * [Jupyter](https://nbviewer.org/github/jrjohansson/scientific-python-lectures/tree/master/):
        Jupyter notebooks with examples of common scientific libraries
        used in Data Science/ML projects (Numpy, Scipy, Matplotlib,...)
        
  * [Anaconda](https://www.anaconda.com) ```#optional``` : open-source
        distribution of Python packages, IDEs, built for data science.

  * [Zeppelin](https://zeppelin.apache.org/) `#optional`: web-based
        notebook that enables data-driven, interactive data analytics
        and collaborative documents with SQL, Scala, Python, R and more.
        
  * [Notebook
        Alternative](https://pbpython.com/notebook-alternative.html)
        `#optional`: running Jupyter Notebook on VS Code
        
* Scientific libraries in Python
  * [Numpy](https://cs231n.github.io/python-numpy-tutorial/):
        scientific library in Python that provides high-performance
        multi-dimensional array and tools for working with them.
        
  * [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html):
        open-source Python package useful for the exploration, cleaning
        and processing of tabular data, called `DataFrame`.
        
  * [Scipy](https://docs.scipy.org/doc/scipy/tutorial/index.html#user-guide):
        collection of mathematical algorithms and convenience functions
        built on `Numpy` (useful for Linear Algebra, Signal Processing,
        Fourier Transforms, Statistics,...)
        
  *  [Scikit-learn
        video](https://www.youtube.com/watch?v=0B5eIE_1vpU) -
        [Scikit-learn guide](https://scikit-learn.org/stable/): machine
        learning library for Python programming language
        
  * [Matplotlib video
        tutorial](https://www.youtube.com/watch?v=6rKe2IEIu8c) -
        [Matplotlib
        guide](https://matplotlib.org/stable/tutorials/introductory/pyplot.html):
        Python library for Data Visualization.
        
  * [Seaborn](https://www.kaggle.com/code/mukeshchoudhary/seaborn-crash-course-a-complete-course-on-seaborn/notebook):
        Python library for data visualization based on `Matplotlib`.
        
  * [Dask-ML](https://ml.dask.org/) ```#optional```: ML library based on
        `Dask` (library for parallel computing in Python).
        
  * [Apache Spark](https://spark.apache.org/) ```#optional```: Fast and
        general engine for large-scale data processing (in batches and
        real-time straming). Spark provides an interface for programming
        clusters (Distributed Computing) with implicit data parallelism
        and fault tolerance.
        
  * [Spark & Hadoop Developer](https://gmucciolo.it/spark-and-hadoop-developer/) ```#optional```
  
  * [Statsmodels](https://www.statsmodels.org/stable/index.html) ```#optional```

<a href="#table-of-contents">🠥🠥 Back to Table of Contents 🠥🠥</a>

## Machine Learning
* [Elements of AI](https://www.elementsofai.com/): An Introduction to AI with no complicated math or programming required.
* [Google Crash Course](https://developers.google.com/machine-learning/crash-course)
* [Book: Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.it/dp/B07XGF2G87/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)
* [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) ```#optional```
* Blog & Articles:
  * [Workflow of a Machine Learning project](https://towardsdatascience.com/workflow-of-a-machine-learning-project-ec1dba419b94)
      Gathering data $\rightarrow$ Data pre-processing $\rightarrow$
        choose the model $\rightarrow$ train and test model
        $\rightarrow$ Evaluation
  * [Machine Learning for Everyone](https://vas3k.com/blog/machine_learning/):  High level
        explanation of ML topics (e.g. Supervised, Unsupervised Learning
        algorithms, Neural networks,... )
  * [Introduction to Data Visualization in Python](https://towardsdatascience.com/introduction-to-data-visualization-in-python-89a54c97fbed): Overview of most common tools for data visualization
        (Matplotlib, Seaborn,Plotly, Pandas Visualization)
  * [Machine learning in Python with Scikit-learn](https://www.dglencross.com/machine%20learning/machine-learning/): Binary Classification example using scikit-learn Random Forest classifier
* **Algorithms**
  * **Supervised Learning**: when each training observation from the dataset has a corresponding label or output value associated with it.
    * [Naive Bayes](https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html), [Support Vector Machine](https://datascience.foundation/datatalk/basic-overview-of-svm-algorithm)
    * Tree-based models:
      * [Random Forest](https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/), [AdaBoost](https://www.mygreatlearning.com/blog/adaboost-algorithm/),[Gradient Boosting](https://blog.mlreview.com/gradient-boosting-from-scratch-1e317ae4587d), [XGBoost](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/), [CatBoost](https://dataaspirant.com/catboost-algorithm/)
      * [Bagging Classifier](https://vitalflux.com/bagging-classifier-python-code-example/), [Voting Classifier](https://towardsdatascience.com/how-voting-classifiers-work-f1c8e41d30ff), [Stacking Classifier](https://bush-dev.com/introduction-to-stacking-classifier/)
      * [K-NN](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)
  
  * **Unsupervised Learning**: when the training data has no labels.
     * [K-Means Clustering](https://www.kdnuggets.com/2019/05/guide-k-means-clustering-algorithm.html)

  * [SageMaker Built-in](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)
        : built-in machine learning algorithms provided by SageMaker
  * [SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)

* **Evaluation Metrics**
  * Classification task: the value of the target variable to predict is discrete
  
    * [Confusion Matrix](https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/), [Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html), [Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score), [Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score),[F Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
    * [ROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html), [ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score), [Hamming Loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss), [Top K Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html#sklearn.metrics.top_k_accuracy_score)
  
  * Regression task: the value of the target variable to predict is continuous
    
    * [MAE](https://www.statisticshowto.com/absolute-error/), [MSE](https://www.freecodecamp.org/news/machine-learning-mean-squared-error-regression-line-c7dde9a26b93/)
 
  * Information Retrieval System (e.g. Recommendation Systems)
        
    * [MRR](https://it.wikipedia.org/wiki/Mean_reciprocal_rank), [DCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain), [NDCG](https://towardsdatascience.com/normalized-discounted-cumulative-gain-37e6f75090e9)
  
  * Images
        
    * [PSNR](https://towardsdatascience.com/deep-learning-image-enhancement-insights-on-loss-function-engineering-f57ccbb585d7), [SSIM](https://towardsdatascience.com/image-classification-using-ssim-34e549ec6e12), [IoU](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
        
* [Homeworks](https://nbviewer.org/github/cs109/2014/tree/master/homework/) ```#optional```
* Extra:
  * [Machine Learning, Data Science and Deep Learning with Python](https://www.udemy.com/course/data-science-and-machine-learning-with-python-hands-on/) ```#optional```
  * [Standardization in case of real-time predictions](https://www.element61.be/en/resource/standardization-case-real-time-predictions) ```#optional```

<a href="#table-of-contents">🠥🠥 Back to Table of Contents 🠥🠥</a>

## Deep Learning
* [Dive into Deep Learning](https://d2l.ai/)

* [Practical Deep Learning for Coders](https://course.fast.ai/)

* [Activation Functions](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6) 
   Functions that multiplie the output of a neuron in a Neural Network. Used to apply a desired transformation to the output.

* [Linear
    Layer](https://medium.com/datathings/linear-layers-explained-in-a-simple-way-2319a9c2d1aa)

* [CNN (Convolutional Neural Networks)](https://cs231n.github.io/): NNs commonly applied to analyze images.

* [RNN (Recurrent Neural Networks)](https://builtin.com/data-science/recurrent-neural-networks-and-lstm):
    NNs commonly applied to analyze Sequential Data (e.g. text,
    audio,...)

* [Optimization](https://d2l.ai/chapter_optimization/): Optimization
    algorithms continuously update model parameters by minimizing the
    value of the loss function. There are different types of
    optimization tools: Gradient descent, SGD, Adagrad, Adam, and so on.

* [Loss Functions / Objective
    Functions](https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23):
    Define an objective which the performance of the model is evaluated
    against and the parameters learned by the model are determined by
    minimizing the chosen loss function.

* [Dropout](https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/dropout_layer):
    Regularization method used to reduce the overfitting issue of large
    neural nets. During training, some number of layer outputs (nodes)
    are "dropped out". This method approximates training a large number
    of neural networks with different architectures in parallel.

* [Batchnorm](https://www.baeldung.com/cs/batch-normalization-cnn):
    Regularization technique used to avoid overfitting and moreover
    improves the learning speed of NN. It normalizes neuron's output
    before applying the activation function.
* [Learning Rate Scheduler](https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1)
* Frameworks:
  * [MXNet](https://mxnet.apache.org/)
  * [PyTorch](https://pytorch.org/)
  * [Tensorflow](https://www.tensorflow.org/)
  * [Keras](https://keras.io) 

<a href="#table-of-contents">🠥🠥 Back to Table of Contents 🠥🠥</a>

## Cloud Operation and DevOps
* [Machine Learning
    Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/machine-learning-lens.html):

    Well-architected framework helps to learn operational and
    architectural best practices for designing and operating ML workflow
    in the cloud.

* [Book: Data Science on
    AWS](https://www.amazon.it/dp/B0921MXC9S/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)

* **AWS Machine Learning Stack**

  * [ML Frameworks & Infrastructure](https://aws.amazon.com/it/machine-learning/infrastructure/)
      AWS services, framework and resources to build, train, and deploy machine learning (ML) applications.

  * [Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
    Fully managed ML service used to quickly and easily build and train ML models and then deploy them into a prediction-ready hosted environment at any scale.

  * [AI Services](https://aws.amazon.com/machine-learning/ai-services/?nc1=h_ls)
    AWS provides several AI services; e.g. Amazon Rekognition that consists in pre-trained and customizable computer vision (CV) capabilities to extract information and insights from your images and videos...

* **More AWS Services**:

  * [AWS Lambda](http://docs.aws.amazon.com/lambda/latest/dg/lambda-introduction.html)
    Serverless computing service that lets you run code in highly available infrastructure without provisioning or managing servers.

  * [CI/CD CodePipeline](https://aws.amazon.com/codepipeline/getting-started/?nc=sn&loc=4)
    Continuous delivery service you can use to model, visualize, and automate the steps required to release your software.

  * [Step Functions](https://docs.aws.amazon.com/step-functions/latest/dg/sample-train-model.html)
    Serverless orchestration service that lets you combine AWS Lambda functions and other AWS services to build business-critical applications. With Step Functions you examine the state of each step in your workflow to make sure that your application runs in order and as expected.

  * [Elastic File System](https://docs.aws.amazon.com/efs/latest/ug/whatisefs.html)
    Serverless elastic file system for use with AWS Cloud services and on-premises resources.

  * [Fargate](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)
    Service that provisions serverless compute resources to run AWS ECS and EKS containers.

  * [AWS Batch](https://docs.aws.amazon.com/batch/latest/userguide/what-is-batch.html)
    Helps you to run batch computing workloads (way to access large amounts of compute resources) on the AWS Cloud across multiple Availability Zones within a Region.

* **Utils**:

  * [Book: Python for DevOps](https://www.amazon.it/dp/B082P97LDW/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1):
    suggested *book* regarding how to use Python for everyday Linux systems administration tasks with today's most useful DevOps tools, including Docker, Kubernetes, and Terraform.

  * [Boto3](https://github.com/boto/boto3) : AWS SDK for Python to create, configure, and manage AWS services, such as Amazon EC2 and Amazon S3.

  * [CloudFormation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/Welcome.html)
    Easy way to create a collection of related AWS resources and provision them in an orderly and predictable fashion. Allows you to model your entire infrastructure in a text file (JSON or YAML) called a **template**.

  * [SAM](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html)
    Framework for building serverless application. It provides shorthand syntax to express functions, APIs, databases, and event source mappings. You can define the application you want and model it using a JSON or YAML configuration template.

  * [Data Science SDK](https://docs.aws.amazon.com/it_it/step-functions/latest/dg/concepts-python-sdk.html)
    With this library you can create workflows that process and publish machine learning models using SageMaker and Step Functions. Data Science SDK provides a Python API that can create and invoke Step Functions workflows.

  * [Scala Data Quality](https://github.com/awslabs/deequ) \| [Python Data Quality](https://github.com/awslabs/python-deequ)
    Python API for Deequ, a library built on top of Apache Spark for defining "unit tests for data", which measure data quality in large datasets. 

  * [Book: Terraform](https://www.amazon.it/Terraform-Running-Writing-Infrastructure-Code/dp/1492046906/) ```#optional```
    Hands-on book exploring Terraform, an Infrastructure as code tool for defining, launching, and managing infrastructure as code (IaC) across a variety of cloud and virtualization platforms.

  * [Troposhpere](https://github.com/cloudtools/troposphere) ```#optional```
    Library that makes easier the creation of the AWS CloudFormation
        JSON by writing Python code to describe the AWS resources.

  * [Goformation](https://github.com/awslabs/goformation) ```#optional```
    Go library for working with AWS CloudFormation / AWS Serverless Application Model (SAM) templates.

  * [Aws Data Wrangle](https://github.com/awslabs/aws-data-wrangler) ```#optional``` :
    Extends the power of Pandas library to AWS connecting DataFrames and AWS data related services.
* Training and Certification
  * [AWS Technical Essentials](https://aws.amazon.com/it/training/course-descriptions/essentials/)
  * [Machine Learning on AWS](https://aws.amazon.com/it/training/learning-paths/machine-learning/data-scientist/)
  * [Data Analytics](https://aws.amazon.com/it/training/path-data-analytics)
* Competitive: [Starway to Orione Cloud](https://github.com/xpeppers/starway-to-orione-cloud)

<a href="#table-of-contents">🠥🠥 Back to Table of Contents 🠥🠥</a>

## Machine Learning Operations (MLOps)
* [What is MLOps](https://www.xenonstack.com/blog/mlops/)
* [MLOps Overview](https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#top_of_page)
* Feature Store:
  * [What is a Feature Store](https://medium.com/data-for-ai/what-is-a-feature-store-for-ml-29b62580af5d)
  * [MLOps with a Feature Store](https://towardsdatascience.com/mlops-with-a-feature-store-816cfa5966e9)
  * [Why Feature Stores They Critical for Scaling Data Science](https://towardsdatascience.com/what-are-feature-stores-and-why-are-they-critical-for-scaling-data-science-3f9156f7ab4)
* [Version Control System for Machine Learning](https://dvc.org/)
* [SageMaker MLOps](https://aws.amazon.com/it/sagemaker/mlops/)
* DevOps for Machine Learning
  * [Automate MLOps with SageMaker Projects](https://www.youtube.com/watch?v=3_cHnk9VSfQ)
  * [Create ML workflows with Amazon SageMaker Pipelines](https://www.youtube.com/watch?v=W7uabCTfLrg)
  * [A/B testing ML with Amazon SageMaker](https://www.youtube.com/watch?v=5WysxAYDH1k)
* Workshop:
  * [MLOps with SageMaker](https://github.com/awslabs/amazon-sagemaker-mlops-workshop)
* Competitive: [Awesome MLOps](https://github.com/visenger/awesome-mlops)

<a href="#table-of-contents">🠥🠥 Back to Table of Contents 🠥🠥</a>

# Machine Learning Applications
A more specific section about some Machine Learning Fields.
## NLP & NLU
* [Introduction to NLP](https://github.com/fastai/course-nlp)
* [Reading Comprehension](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2718305.pdf)
* [Question Answering on the SQuAD Dataset](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2761899.pdf)
* [BERT Explained](https://towardsml.wordpress.com/2019/09/17/bert-explained-a-complete-guide-with-theory-and-tutorial/)
* [Transformers](https://huggingface.co/course/chapter1/1)
* [SpaCy](https://course.spacy.io/en/)
* [Learning to Rank](https://www.amazon.it/Learning-Rank-Information-Retrieval-Tie-yan/dp/3642142664/)
* [Natural Language Processing with Python](https://www.nltk.org/book/)
* [Unsupervised Translation of Programming Languages](http://www.iitp.ac.in/~ai-nlp-ml/course/dnlp/cep_iitpatna_mt_2020.pdf)
* [Perplexity](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/) and [BLEU Score](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/) metrics

<a href="#table-of-contents">🠥🠥 Back to Table of Contents 🠥🠥</a>

## Recommendation System
* [What Are Recommender Systems](https://builtin.com/data-science/recommender-systems)
* [Matrix Factorization](https://medium.com/@paritosh_30025/recommendation-using-matrix-factorization-5223a8ee1f4) 
* [How does Netflix recommend movies using Matrix Factorization](https://www.youtube.com/watch?v=ZspR5PZemcs&t=653s)
* [Collaborative filtering](https://www.youtube.com/watch?v=h9gpufJFF-0)

<a href="#table-of-contents">🠥🠥 Back to Table of Contents 🠥🠥</a>

## Reinforcement Learning
* [Reinforcement Learning Introduction](https://www.udacity.com/course/reinforcement-learning--ud600)
* Videos:
  * [Getting started with OpenAI Gym](https://www.youtube.com/watch?v=8MC3y7ASoPs)
  * [Q-Learning](https://www.youtube.com/watch?v=qhRNvCVVJaA)
  * [Q Learning Intro/Table](https://www.youtube.com/watch?v=yMk_XtIEzH8)
  * [Q Learning Algorithm and Agent](https://www.youtube.com/watch?v=Gq1Azv_B4-4)
  * [Q-Learning Agent Analysis](https://www.youtube.com/watch?v=CBTbifYx6a8)
  * [Creating A Reinforcement Learning (RL) Environment](https://www.youtube.com/watch?v=G92TF4xYQcU)
  * [Deep Q Learning w/ DQN](https://www.youtube.com/watch?v=t3fbETsIBCY)
  * [Training & Testing Deep reinforcement learning (DQN) Agent](https://www.youtube.com/watch?v=qfovbG84EBg)
* Reinforcement Learning on AWS:
  * [SageMaker DeepRacer](https://aws.amazon.com/it/deepracer/)
  * [Deep Reinforcement Learning Models: Tips & Tricks for Writing Reward Functions](https://medium.com/@BonsaiAI/deep-reinforcement-learning-models-tips-tricks-for-writing-reward-functions-a84fe525e8e0)
  * [Proximal Policy Optimization](https://openai.com/blog/openai-baselines-ppo/) ```#optional```

<a href="#table-of-contents">🠥🠥 Back to Table of Contents 🠥🠥</a>
