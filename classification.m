%% Chapter 6: Document Filtering (Page 117)
% "Programming Collective Intelligence - Building Smart Web 2.0 Applications" 
% by Toby Segaran (O'Reilly Media, ISBN-10: 0-596-52932-5)
%
% This chapter shows how to classify documents based on their contents as 
% an application of machine learning. A familiar application of such 
% techniques is spam filtering. This problem began with e-mails, but it has
% now spread to blogs, message boards, Wiki's, and social networking 
% sites like MySpace and Facebook in a form of spam posts, comments and 
% messages. This is the dark side of the Web 2.0.
%
% However, the techniques are not limited to spam for application. The
% algorithms are broad enough to accommodate other applications, because it
% is about learning to recognize whether a document belongs in one category
% or another based on its content.
%
% In the world of marketing I live in, we also gets a lot of junk leads as
% we move our lead generation activities to online. We would use online
% forms to try to gather data, but your target may try to bypass the forms
% by providing fake data in order to get the fulfillment. It would be very
% useful to automatically sort out the junk. 
%
% In this chapter I will also take advantage of a new MATLAB feature in
% R2008a called object-oriented programming (OOP).

%% Documents and Words (Page 118)
%
% The task of classifying documents requires some kind of features to
% distinguish one document over another. A feature is anything that you can
% determine as being either present or absent in the document, and in our
% case words in documents are the logical choice for such features.
% Assumption is that frequency of word appearance is different based
% on document type. For example, certain words are more likely to
% appear in spam than non-spam. 
%
% Of course, features could be word pairs or phrases, or any other higher
% or lower document structures that can be either present or absent. In 
% this case, we will focus on words.
% 
% 'getwords' function extracts words larger than 2 letters but less than 20
% into a cell array using regular expression. The text is divided by
% non-alphabetic character, so this doesn't work with Japanese text. Words
% are converted to lowercase. 
%
% This is not a perfect solution, because spam often uses capitalized
% letters and non-alphabetic characters for emphasis, and we are not
% capturing these features in 'getwords'. So we need to leave room for
% improvement when we write the classification routine by easily
% substituting with different feature extraction techniques.

test='This is just a test %*$![]';
disp(['test sample=''' test ''';'])
disp('Extract features using ''getwords'' function')
disp(getwords(test))
clear all;

%% Training the Classifier (Pages 119-121)
%
% The classifiers discussed in this chapter learn how to classify a
% document by being trained. The more examples of documents and their
% correct classifications it sees, the better the classifier will get at
% making predictions with increasing certainty.
%
% Here, the classifier is implemented as a user-defined MATLAB class,
% taking advantage of the new feature in R2008a. Classes act as templates.
% We may want to classify different documents by different criteria, and we
% can write separate programs for them, but by using template approach, we
% can use one template that can be used to create individual cases of
% classifiers for each situation. These individual cases are called
% 'instances' or 'objects' in OOP. We can create different instances of the
% classifier class or template and train them differently to have them
% serve different purposes. 
%
% Make sure you use 'handle' superclass declaration in class definition 
% - this is the key feature that enables self-referencing behavior for your
% user-defined classes. This key feature is not well documented in MATLAB 
% documentation, so it took me a while to figure this out.
%
% 'classifier' class has 3 built-in properties. 'fc' stands for 'feature
% count' and it keeps track of different features in different categories
% of classification. 'cc' stands for 'category count' and it keeps track of
% how many times each category has been used. The last property,
% 'getfeatures', store the function handle to a feature-extraction
% function. Default is '@getwords'. The class also defines the methods you 
% can use to construct an object or instance, or manipulate it in various 
% ways. 'train' and 'fcount' are examples of those methods.
%
% 'train' method provides the initial training for classifier object by
% supplying a sample text and its category. 'fcount' gives you the count of
% given feature in a given category acquired through training.

cl=classifier(); % create an instance or object from 'classifier' class
text1='the quick brown fox jumps over the lazy dog';
text2='make quick money in the online casino';
disp(['training sample1=''' text1 ''';'])
disp(['training sample2=''' text2 ''';'])
cl.train(text1,'good'); % train the classifier object 'cl' with sample text.
cl.train(text2,'bad'); % more training, with a bad example. 
% the word 'quick appeared once on both, so it should have a count of 1
% in each category
disp(' ')
disp(sprintf('Count of ''quick'' classifed as ''good''=%d',cl.fcount('quick', 'good')))
disp(sprintf('Count of ''quick'' classifed as ''bad''=%d',cl.fcount('quick', 'bad')))

%% Calculating Probabilities (Pages 121-122)
%
% Now that we know how many features a given category has and frequency of
% each feature, you can now calculate the probability of a word being in a
% particular category... Pr(word|category). This is implemented as a class
% method 'fprob'. 
%
% We also have to re-train the object as we develop the algorithm, so we
% should automate the training process - that's what 'sampletrain' does.
%
% The sample texts it feeds are:
%
% * 'Nobody owns the water.' -> good
% * 'the quick rabbit jumps fences' ->good
% * 'buy pharmaceuticals now' -> bad
% * 'make quick money at the online casino' -> bad
% * 'the quick brown fox jumps' -> good
%
% 'quick' appears 3 times and 2 are good and 1 bad. So the probability of
% 'quick' appearing in 'good' documents should be 2/3=0.6666666..

disp(' ')
disp('Reset the ''classifier'' object')
cl=classifier(); % reset the object by re-creating it.
sampletrain(cl); % feed the standard training data
format='Probability of ''quick'' appearing in ''good'' documents=%f';
disp(sprintf(format,cl.fprob('quick','good')))

%% Starting with a Reasonable Guess (Pages 122-123)
%
% 'fprob' gives exact prediction based on training data, but that's also
% its limitation - its prediction is good only for training data. Its
% prediction is incredibly sensitive to the training data it received.
% For example, the word 'money' only appeared once in a bad example, so its
% probability is 100% for bad documents, 0% for good documents. This is too
% extreme, because 'money' could be used in good documents as well.
%
% To address this issue, we would like to introduce a concept of 'assumed
% probability' ('ap') that new features take on initially, and then adjusted
% gradually in training. '0.5' is a good neutral probability to start with.
% Next thing to think about is how much weight you would like to give to 'ap'.
% Weight of 1 means it is worth one word in actual samples. The new metric
% is the weighted average of probability returned from 'fprob' and 'ap'.
%
% For the case of 'money', its 'ap' is '0.5' initially for both 'good' and  
% 'bad'. Then the classifier object finds 1 example of 'money' in a 'bad' 
% document in the training. So its probability for 'bad' becomes:
%
% * (weight x ap + count x fprob)/(count + weight)
% * = (1x0.5+1*1)/(1+1)
% * = 0.75
% 
% This means that its probability for 'good' will be '0.25'.

disp(' ')
disp('Reset the ''classifier'' object')
cl=classifier();
sampletrain(cl);
format='Probability of ''money'' appearing in ''good'' documents=%f';
disp('Run the weighted probability calculation... ')
disp(' ')
disp(sprintf(format,cl.weightedprob('money','good',@cl.fprob)))
disp(' ')
disp('Repeat the training - this should increase certainty')
disp(' ')
sampletrain(cl);
disp(sprintf(format,cl.weightedprob('money','good',@cl.fprob)))

%% A Naive Classifier - Probability of a Whole Document (Pages 123-124)
%
% We now have a way to calculate Pr(word|category), which gives us the 
% probabilities of a document in a category containing a particular word,
% but we need a way to combine the individual word probabilities to get the 
% probability that an entire document belongs in a given category.
%
% Here we will implement one approach called a naive Bayesian classifier. 
% It is called 'naive' because it assumed that the probabilities being
% combined are independent of each other. In the case of spam, we know that
% 'money' and 'casino' are much more highly correlated than other
% combinations, so this is not a good assumption. While we cannot use the
% probabilities given by a naive Bayesian classifier per se, we can still 
% use its results relatively for comparison among different categories.
% Therefore it is still an effective approach in real life despite its flaw
% in the assumption. 
%
% If all features are independent, then we can combine their probabilities
% by multiplying them together. Let's take an example of words 'MATLAB' and
% 'Casino', and their probabilities to appear in bad documents are as 
% follows:
% 
% * Pr(matlab|bad)=20%
% * Pr(casino|bad)=80%
% 
% The probability that both 'MATLAB' and 'Casino' should appear in bad
% documents is 16%. 
%
% * Pr(matlab&casino|bad)= 20%x80%=16%
%
% So we can get the probabilities for the whole document by multiplying the
% probabilities of all features contained in the document.
%
% We will create a different subclass of 'classifier' to add this
% capability. This subclass is called 'naivebayes' which implements
% 'docprob' method.
%
% Let's test with some sample data. Please remember that the result won't
% be exactly 16% because we are using weighted average here. 

disp(' ')
disp('Reset the object with ''naivebayes''')
cl=naivebayes();
disp(' ')
disp('Training ''naivebayes'' object with 80% good ''MATLAB'' & 80% bad ''Casino''...')
cl.fc={'matlab','good',80;'matlab','bad',20; 'casino','good',20;'casino','bad',80};
cl.cc={'good',100;'bad',100};
format='Probability of ''matlab casino'' as a ''good'' document=%f';
disp(sprintf(format,cl.docprob('matlab casino','good')))

%% A Quick Introduction to Bayes' Theorem (Pages 125-126)
%
% Now that we know how to calculate Pr(document|category), we need to flip
% it around to get Pr(category|document) in order to classify new
% documents. That's what Bayes' Theorem is about.
%
% * Pr(A|B)=Pr(B|A)xPr(A)/Pr(B)
%
% Or, in our case,
%
% * Pr(category|document)=Pr(document|category)xPr(category)/Pr(document)
% 
% Pr(document|category) is given by 'docprob'. Pr(category) is just the
% number of documents in the category divided by total number of documents.
% Pr(document) is irrelevant in our case. Because of the basic flaw in our
% naive Bayesian approach, we are not going to use resulting probabilities
% directly. Instead we will calculate probability for each category 
% separately, and then all the results will be compared. Pr(document) is 
% the same for each of those separate calculations, so we can safely ignore 
% this term.
%
% 'prob' method calculates the probability of each category and returns the
% product Pr(document|category) x Pr(category). 

disp(' ')
disp('Reset the ''naivebayes'' object')
cl=naivebayes();
sampletrain(cl);
format='Probability of ''quick rabbit'' as a ''good'' document=%f';
disp(sprintf(format,cl.prob('quick rabbit','good')))
format='Probability of ''quick rabbit'' as a ''bad'' document=%f';
disp(sprintf(format,cl.prob('quick rabbit','bad')))

%% Choosing a Category (Pages 126-127)
%
% The final step in classification is to determine the category a new
% document belongs in. The simple approach is to assign it to the category
% with highest probability. However, in reality, this may not be a
% desirable behavior. You don't want to misplace non-spam as spam just
% because the probability is marginally higher. So we need to assign a
% minimum threshold for each category, so that new item can be placed there
% only when the probability is higher than the next best by a specified
% threshold. 
%
% For example, the threshold for 'bad' could be set to '3', while it could
% be '1' for 'good'. This means any item with highest 'good' probability,
% however small the difference may be, will be classified as 'good', while
% only items with 3 times higher 'bad' probability than others will be 
% classified as 'bad', and anything inbetween is 'unknown'.
%
% We need to modify 'classifier' class to add 'thresholds' property and
% 'classify' method to implement this approach. The changes will also be 
% inherited by 'naivebayes' subclass automatically, so you don't have to do
% anything with that subclass. This is the power of OOP.

disp(' ')
disp('Reset the ''naivebayes'' object')
cl=naivebayes();
sampletrain(cl);
format='Bayes classify ''quick rabbit''... Category=%s';
disp(sprintf(format,cl.classify('quick rabbit','unknown')))
format='Bayes classify ''quick money''... Category=%s';
disp(sprintf(format,cl.classify('quick money','unknown')))
disp(' ')
disp('Set the threshold for ''bad'' to ''3.0''')
cl.setthreshold('bad',3.0);
disp(sprintf(format,cl.classify('quick money','unknown')))
disp(' ')
disp('Run the ''sampletrain'' 10 times')
for i=1:10
    sampletrain(cl);
end
disp(sprintf(format,cl.classify('quick money','unknown')))
clear all;

%% The Fisher Method - Category Probabilities for Features (Pages 127-129)
%
% Here we will examine the Fisher Method, named for R.A. Fisher, as an
% alternative approach in document filtering. This method is used in
% SpamBayes, an Outlook plugin written in Python, and known to be
% particularly effective for spam filtering.
% 
% In a naive Bayesian approach, we used the feature probabilities to
% determine the whole document probability. The Fisher Method is more
% elaborate, but it is worth learning because it offers much greater
% flexibility.
%
% In the Fisher Method, you start by calculating the probability of a
% document being in a category given the presence of a particular feature -
% Pr(category|feature). If 'casino' appears in 500 documents and 499 of
% those are in the 'bad' category, then 'casino' will score very close to 1
% for 'bad'.
%
% * Pr(category|feature)= (number of documents in this category with the
% feature)/(total number of documents with the feature)
%
% However, this formula doesn't account for imbalance of data among
% different categories. If most of your documents are good except a few bad
% ones, the features associated with bad ones will score too high for
% 'bad', even though they may be perfectly legitimate words. So we need to
% normalize the probability calculation. Required information is: 
%
% * clf=Pr(feature|category) for a given category
% * freqsum = sum of Pr(feature|category) for all the categories
% * cprob=clf/(clf+nclf)
%
% If you run the following code, you will see that 'money' returns 1.0
% probability for 'bad'. Again, it is too sensitive to the training data. 
% We should use weighted average approach again with the Fisher Method.

disp(' ')
disp('Reset the object with ''fisherclassifier''')
cl=fisherclassifier();
sampletrain(cl);
format='Probability of ''quick'' as a ''good'' document=%f';
disp(sprintf(format,cl.cprob('quick','good')))
format='Probability of ''money'' as a ''bad'' document=%f';
disp(sprintf(format,cl.cprob('money','bad')))
disp(' ')
disp('Use weighted average rather than cprob directly')
disp(sprintf(format,cl.weightedprob('money','bad',@cl.cprob)))

%% Combining the Probabilities (Pages 129-130)
%
% In our naive Bayesian approach we combined the feature probabilities by 
% multiplying them to derive an overall document probability. The Fisher
% Method add rigor to this process by applying additional steps to test the
% multiplication result.
%
% Fisher's research shows that if you take a natural log of the result and
% multiplying it by -2, then the resulting score should fit a chi-squared
% distribution if the probabilities are independent and random. We can use
% this knowledge this way: if a document doesn't belong in a particular 
% category, its feature probabilities vary randomly for that category.
% If a document belongs in another category, its feature probabilities
% should be high in general - no longer random. 
% 
% By feeding the result of the Fisher calculation to the inverse chi-square 
% function, you get the probability that a random set of probabilities 
% would return such a high number. For further reading, check out:
% http://www.gigamonkeys.com/book/practical-a-spam-filter.html
%
% By the way, I made a small change to 'invchi2' function: 
%
% * Python: for i in range(1, df//2)
% * MATLAB: for i=1:floor(df/2)-1
%
% '//' is an integer division operator, an equivalent to 'floor' in MATLAB. 
% 'range(1,2)' returns '[1]', not [1 2] in Python, however - the second 
% end-of-range argument is not included in the resulting vector. So I had 
% to add '-1' in order to make MATLAB version to behave the same as Python 
% version. I am not sure if this is strictly Kosher from  Statistics point 
% of view. 
% 
% Anyhow, we can see that the Fisher Method is more sophisticated and
% robust than our naive Bayesian alternative, and therefore it is expected 
% to perform better.   

disp(' ')
disp('Reset the ''fisherclassifier'' object')
cl=fisherclassifier();
sampletrain(cl);
format='''cprob'' of ''quick'' as a ''good'' document=%f';
disp(sprintf(format,cl.cprob('quick','good')))
format='''fisherprob'' of ''quick rabbit'' as a ''good'' document=%f';
disp(sprintf(format,cl.fisherprob('quick rabbit','good')))
format='''fisherprob'' of ''quick rabbit'' as a ''bad'' document=%f';
disp(sprintf(format,cl.fisherprob('quick rabbit','bad')))

%% Classifying Items (Pages 130-131)
%
% Because 'fisherprob' returns a probability value pre-qualified by inverse
% chi-square function, we can use it directly in classification, rather
% than having multiplication thresholds like our naive Bayesian classifier.
% Instead, we will specify the minimums for each classification. For a
% spam filter, you may want to set this minimum high for 'bad', such as '0.6', 
% while the minimum for 'good' can be as low as '0.2', in order to avoid
% accidental misplacement of good e-mails in the 'bad' category. Anything
% lower than the minimums will be classified as 'unknown'.
% 
% The result we will see may be not much different from the output of our 
% naive Bayesian classifier, because of the small training dataset, but The
% Fisher Method is believed to outperform in real world situations. 

disp(' ')
disp('Reset the ''fisherclassifier'' object')
cl=fisherclassifier();
sampletrain(cl);
format='Fisher classify ''quick rabbit''... Category=%s';
disp(sprintf(format,cl.classify('quick rabbit','unknown')))
format='Fisher classify ''quick money''... Category=%s';
disp(sprintf(format,cl.classify('quick money','unknown')))
disp(' ')
disp('Set the minumum for ''bad'' to ''0.8''')
cl.setminimum('bad',0.8);
disp(sprintf(format,cl.classify('quick money','unknown')))
disp(' ')
disp('Set the minumum for ''good'' to ''0.4''')
cl.setminimum('good',0.4);
disp(sprintf(format,cl.classify('quick money','unknown')))

%% Persisting the Trained Classifiers (Pages 132-133)
%
% The book covers the process of adapting the 'classifier' class to work
% with SQLite light-weight, serverless database. Python 2.5 or later comes
% with SQlite built-in. It is used widely in desktop applications as well
% as small embedded applications. See http://www.sqlite.org/ for more
% information. 
%
% So far, the classifiers we created need to be trained each time before
% you can use them, and you need to repeat this process each time you start
% up a session in MATLAB, because we haven't provided a way to persist the
% data or object. In MATLAB, you can actually save the whole classifier
% objects in .mat file. Python doesn't provide such functionality built-in,
% so it makes sense to provide native support to SQLite precisely for this
% reason. 
% 
% There is a nice library called 
% <http://mksqlite.berlios.de/mksqlite_eng.html mksqlite> that connects
% MATLAB to SQLite.
% "classifer2.m" shows the modified classifier that uses mksqlite to
% persist the trained object.
%
% Here is the simple script to test the modified classifier. 

% text1='the quick brown fox jumps over the lazy dog';
% text2='make quick money in the online casino';
%
% cl=classifier2();
% cl.setdb('test1.db');
% cl.train(text1, 'good');
% cl.train(text2, 'bad');
%
% disp(sprintf('Count of ''quick'' classified as ''good''=%d', cl.fcount('quick', 'good')))
% disp(sprintf('Count of ''quick'' classified as ''bad''=%d,   cl.fcount('quick', 'bad')))
% mksqlite(cl.con, 'close');
%
% clear all

%% Filtering Blog Feeds - interactive training (Pages 134-135)
%
% As a non-spam related application of the classifiers we created, the book
% introduces a program that filters RSS feeds from various blogs. RSS was
% one of the important drivers for Web 2.0, but it now floods the
% cyberspace, overwhelming people with too much information to process.
%
% It takes advantage of a RSS parsing library called 'Universal Feed
% Parser' to build the documents to feed to the classifier, present the
% text to the user with a guess, then asks for a user input for correct
% answer. You can pick from multiple categories beyond 'good' or 'bad'.
% This user input is then used for training. 
%
% Since there is no 'Universal Feed Parser' for MATLAB, I wrote a very
% simple code inside 'feedfilter' function. It only works with the sample
% XML file provided here 'python_search.xml' or any RSS feed that follows
% its format exactly. It doesn't work with other RSS feed formats. 
%
% I improved the usability a bit - you can simply accept a guess by hitting
% return without entering text. You can also end the session by entering
% 'end' at the prompt at any time. As you go through the training, you will
% notice that the guesses get better and better. 

% Test 'feedfilter' function 
% cl=fisherclassifier();
% feedfilter(cl);

%% Filtering Blog Feeds - the training results (Page 136)
%
% If you would rather skip the tedious interactive training, you can load
% the pre-made dataset. The probabilities of various categories given the
% word 'python' are evenly divided, because this word appears in all
% entries. The word 'Eric' occurs 25 percent of entries related to Monty
% Python and doesn't occur at all in other categories. Therefore the
% probability of 'monty' given 'eric is 100%, while the probability of
% 'eric' given 'monty' is 25% (well, in the way I classified, it comes out 
% more like 28%). 

load feedfilter.mat;
disp(' ')
format='Probability (w->c) of ''python'' in ''prog'' = %f';
disp(sprintf(format,cl.cprob('python','prog')))
format='Probability (w->c) of ''python'' in ''snake'' = %f';
disp(sprintf(format,cl.cprob('python','snake')))
format='Probability (w->c) of ''python'' in ''monty'' = %f';
disp(sprintf(format,cl.cprob('python','monty')))
format='Probability (w->c) of ''eric'' in ''monty'' = %f';
disp(sprintf(format,cl.cprob('eric','monty')))
format='Probability (c->w) of ''eric'' in ''monty'' = %f';
disp(sprintf(format,cl.fprob('eric','monty')))
clear all;

%% Improving Feature Detection (Pages 136-138)
%
% Our 'getwords' function is very simple. It ignores all non-alphabet
% characters and convert everything into lowercase. There are many ways to
% improve it:
%
% * Use the presence of many uppercase words as a feature without making
% uppercase words and lowercase words distinct.
% * Use set of words in addition to individual words.
% * Capture more meta information, such as who sent an email message or what
% category a blog entry was posted under, and annotate it as
% meta information.
% * Keep URLs and numbers intact.
%
% 'entryfeatures' works with RSS feed add those capabilities. 'feedfilter'
% is also modified to pass a RSS entry as a cell array function rather than
% as a text string to feature detection if the function name is
% 'entryfeatures'. 

% cl=fisherclassifier(@entryfeatures);
% feedfilter(cl);

%% Using Akismet (Pages 138-139)
%
% Akismet http://akismet.com/ is another web service that provides spam 
% filter function online via its XML-based API. This means that we can take
% advantage of collective power of online users to fight spam, so that you
% don't have to train your spam filter on your own.
% 
% It is primarily designed for filtering out spam comments posted on blogs,
% so it doesn't work for other uses very well. Therefore I am not going to
% touch it here. 
%
% But this is a good example of how Web 2.0 takes advantage of collective
% intelligence. 

%% Alternative Methods (pages 139-140)
%
% The book discusses other approaches such as neural network or support
% vector machines. However, Bayesian approaches we covered here are easier
% to implement and demand less computing power and resources.



