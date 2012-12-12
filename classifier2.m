classdef classifier2 < handle % subclass of 'handle' superclass
% CLASSIFIER defines a class of objects that classifies documents based on 
% their features. Constructor method takes an optional feature extraction 
% function reference.
%   Created to experiment with MATLAB OOP features.
%   Modified to store data in SQLite database using mksqlite

   properties
       % Function to extract features
       getfeatures
       % probability thresholds for classifying into given categories
       thresholds;
       % Database Connection ID
       con;
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
           self.thresholds={};
       end
       
       % Open or create a database 
       function setdb(self,dbfile)
           self.con= mksqlite(0, 'open', dbfile);
           mksqlite(self.con,'create table if not exists fc(feature,category,count)');
           mksqlite(self.con,'create table if not exists cc(category,count)');
       end
       
       % Increase the count of a feature/category pair
       function incf(self,f,cat)
           count=self.fcount(f,cat);
           if count==0
               sq='insert into fc values (';
               sq=[sq sprintf('''%s'',''%s'',1',f,cat) ')'];
               mksqlite(self.con, sq);
           else
               sq='update fc set count=';
               sq=[sq sprintf('%d where feature=''%s'' and category=''%s''',count+1,f,cat)];
               mksqlite(self.con, sq);
           end
       end
      
       % Increase the count of a category
       function incc(self,cat)
           count=self.catcount(cat);
           if count==0
               sq='insert into cc values (';
               sq=[sq sprintf('''%s'',1',cat) ')'];
               mksqlite(self.con, sq);
           else
               sq='update cc set count=';
               sq=[sq sprintf('%d where category=''%s''',count+1,cat)];
               mksqlite(self.con,sq);
           end
       end
    
       % The number of times a feature has appeared in a category
       function c=fcount(self,f,cat)
           sq='select count from fc where feature=';
           sq=[sq sprintf('''%s'' and category=''%s''',f,cat)];
           sq = [sq ' limit 1'];
           res=mksqlite(self.con, sq);
           if isempty(res)
               c=0;
           else
               c=res.count;
           end
       end
       
       % The number of item in a category
       function c=catcount(self,cat)
           sq=sprintf('select count from cc where category=''%s''',cat);
           sq=[sq ' limit 1'];
           res=mksqlite(self.con, sq);
           if isempty(res)
               c=0;
           else
               c=res.count;
           end
       end
       
       % The total number of items
       function c=totalcount(self)
           res=mksqlite(self.con, 'select sum(count) from cc');
           if isempty(res)
               c=0;
           else
               c=res.('sum(count)');
           end
       end
       
       % The list of all categories
       function list=categories(self)
           res=mksqlite(self.con, 'select category from cc');
           list={res.category}';
       end

       % 'train' method takes an item (a document) and a category and
       % extract features using getfeature function, then increment the
       % feature/category counts. 
       function train(self,item,cat)
           features=self.getfeatures(item);
           for f=1:size(features,1)
               self.incf(features{f},cat);
           end
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
