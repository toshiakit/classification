classdef fisherclassifier < classifier
% FISHERCLASSIFIER extend classifier superclass by providing The Fisher
% Method functionalities.
%   Because of inheritance, 'fisherclassifier' will have the same
%   properties and methods with superclass 'classifier'. So we only have to 
%   define unique add-on functionalities in 'fisherclassifier'.
    
   properties
       % cutoff for classifying into given categories
       minimums;
   end
   methods
       % Constructor Method
       function self=fisherclassifier(getfeatures)
           if nargin==0
               self.getfeatures=@getwords;
           else
               self.getfeatures=getfeatures;
           end
           self.minimums={};
       end
       
       function p=cprob(self,f,cat)
           % The frequency of this feature in this category
           clf=self.fprob(f,cat);
           if clf==0
               p=0;
           else
               % The frequency of this feature in all the categoris
               c=self.categories();
               freqsum=0;
               for i=1:size(c,1)
                   freqsum=freqsum+self.fprob(f,c{i});
               end
               % The probability is the frequency in this category divided
               % by the overall frequency
               p=clf/(freqsum);
           end
       end
       
       function fprob=fisherprob(self,item,cat)
           %Multiply all the probabilities together
           p=1;
           features=self.getfeatures(item);
           for i=1:size(features,1)
               p=p*self.weightedprob(features{i},cat,@self.cprob);
           end
           
           % Take the natural log and multiply by -2
           fscore=-2*log(p);
           
           % Use the Chi-square inverse function to get a probability
           fprob=self.invchi2(fscore,size(features,1)*2);
       end
       
       function x=invchi2(self,chi,df)
           m=chi/2.0;
           sum=exp(-m);
           term=sum;
           for i=1:floor(df/2)-1
               term=term*m/i;
               sum=sum+term;
           end
           x=min(sum,1.0);
       end
       
       function setminimum(self,cat,min)
           if isempty(strmatch(cat,char(self.minimums{:,1}), 'exact'))
               self.minimums{end+1,1}=cat;
               self.minimums{end,2}=min;
           else
               idx=strmatch(cat,char(self.minimums{:,1}), 'exact');
               self.minimums{idx,2}=min;
           end
       end
       
       function m=getminimum(self,cat)
           if isempty(strmatch(cat,char(self.minimums{:,1}), 'exact'))
               m=0;
           else
               idx=strmatch(cat,char(self.minimums{:,1}), 'exact');
               m=self.minimums{idx,2};
           end
       end
       
       function best=classify(self,item,default)
           if nargin <3
               default='';
           end
           % Loop through looking for the best result
           best=default;
           max=0.0;
           c=self.categories();
           for i=1:size(c,1)
               p=self.fisherprob(item,c{i});
               % Make sure it exceeds its minimum
               if (p>self.getminimum(c{i}))&&(p>max)
                   best=c{i};
                   max=p;
               end
           end
       end
   end
end 
