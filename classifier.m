classdef classifier < handle % subclass of 'handle' superclass
% CLASSIFIER defines a class of objects that classifies documents based on 
% their features. Constructor method takes an optional feature extraction 
% function reference.
%   Created to experiment with MATLAB OOP features.

   properties
       % Counts of feature/category combinations
       fc;
       % Counts of documents in each category
       cc;
       % Function to extract features
       getfeatures
       % probability thresholds for classifying into given categories
       thresholds;
   end

   % Public Methods
   methods
       % Constructor Method
       function self=classifier(getfeatures)
           if nargin==0
               self.getfeatures=@getwords;
           else
               self.getfeatures=getfeatures;
           end
           self.fc={};
           self.cc={};
           self.thresholds={};
       end
       
       % Increase the count of a feature/category pair
       function incf(self,f,cat)
           if isempty(strmatch(f,char(self.fc{:,1}), 'exact'))
               self.fc{end+1,1}=f;
               self.fc{end,2}=cat;
               self.fc{end,3}=1;
           else
               idx=strmatch(f,char(self.fc{:,1}), 'exact');
               if isempty(strmatch(cat,char(self.fc{idx,2}), 'exact'))
                   self.fc{end+1,1}=f;
                   self.fc{end,2}=cat;
                   self.fc{end,3}=1;
               else
                   idx=idx(strmatch(cat,char(self.fc{idx,2})));
                   self.fc{idx,3}=self.fc{idx,3}+1;
               end
           end
       end
       
       % Increase the count of a category
       function incc(self,cat)
           if isempty(strmatch(cat,char(self.cc{:,1}), 'exact'))
               self.cc{end+1,1}=cat;
               self.cc{end,2}=1;
           else
               idx=strmatch(cat,char(self.cc{:,1}), 'exact');
               self.cc{idx,2}=self.cc{idx,2}+1;
           end
       end
    
       % The number of times a feature has appeared in a category
       function c=fcount(self,f,cat)
           idx=strmatch(f,char(self.fc{:,1}), 'exact');
           if ~isempty(idx) && ~isempty(strmatch(cat,char(self.fc{idx,2}), 'exact'))
               idx=idx(strmatch(cat,char(self.fc{idx,2})));
               c=self.fc{idx,3};
           else
               c=0;
           end
       end
       
       % The number of item in a category
       function c=catcount(self,cat)
           idx=strmatch(cat,char(self.cc{:,1}), 'exact');
           if ~isempty(idx)
               c=self.cc{idx,2};
           else
               c=0;
           end
       end
       
       % The total number of items
       function c=totalcount(self)
           c=sum(cell2mat(self.cc(:,2)));
       end
       
       % The list of all categories
       function list=categories(self)
           if ~isempty(self.cc)
            list=self.cc(:,1);
           else
               list=cell(0,1);
           end
       end

       % 'train' method takes an item (a document) and a category and
       % extract features using getfeature function, then increment the
       % feature/category counts. 
       function train(self,item,cat)
           % Extract features
           features=self.getfeatures(item);
           % Increment the count for every feature with this category
           for i=1:size(features,1)
               self.incf(features{i},cat);
           end
           % Increment the count for this category
           self.incc(cat);
       end
       
       function p=fprob(self,f,cat)
           if self.catcount(cat)==0
               p=0;
           else
               % The total number of times this feature appeared in this
               % category divided by the total number of items in this
               % category... Pr(feature|category)
               p=self.fcount(f,cat)/self.catcount(cat);
           end
       end
       
       % 
       function bp=weightedprob(self,f,cat,prf,weight,ap)
           if nargin <3
               error('Not enough input arguments.')
           elseif nargin <4
               prf=@fprob; % function reference for basic probability
               weight=1.0; % weight of assumed probability
               ap=0.5; % assumed probability, initially 0.5 = neutral
           elseif nargin <5
               weight=1.0;
               ap=0.5;
           elseif nargin <6
               ap=0.5;
           end
               
           % Calculate current probability
           basicprob=prf(f,cat);
           
           % Count the number of times this feature has appeared in all
           % categories
           cat=self.categories();
           totals=0;
           for i=1:size(cat,1)
               totals=totals+self.fcount(f,cat{i});
           end
           
           % Calculate the weighted average
           bp=((weight*ap)+(totals*basicprob))/(weight+totals);
       end
       
       function setthreshold(self,cat,t)
           if isempty(strmatch(cat,char(self.thresholds{:,1}), 'exact'))
               self.thresholds{end+1,1}=cat;
               self.thresholds{end,2}=t;
           else
               idx=strmatch(cat,char(self.thresholds{:,1}), 'exact');
               self.thresholds{idx,2}=t;
           end
       end
       
       function t=getthreshold(self,cat)
           if isempty(strmatch(cat,char(self.thresholds{:,1}), 'exact'))
               t=1;
           else
               idx=strmatch(cat,char(self.thresholds{:,1}), 'exact');
               t=self.thresholds{idx,2};
           end
       end
       
       function category=classify(self,item,default)
           if nargin <3
               default='';
           end
           % Find the category with the highest probability
           max=0.0;
           cat=self.categories();
           probs=zeros(size(cat,1),1);
           for i=1:size(cat,1)
               probs(i,1)=self.prob(item,cat{i});
               if probs(i,1)>max
                   max=probs(i,1);
                   best=cat{i};
               end
           end
           % Make sure the probability exceeds threshold*next best
           for i=1:size(cat,1)
               if strcmp(cat{i},best)
                   continue;
               end
               if probs(i,1)*self.getthreshold(best)>probs(strmatch(best,char(cat),'exact'))
                   category=default;
               else
                   category=best;
               end
           end   
       end
   end
end
