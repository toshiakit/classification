classdef naivebayes < classifier
% NAIVEBAYES extend classifier superclass by providing method to calculate
% the probabilities for the whole document Pr(document|category) by
% combining the probabilities of features in the document. 
%   Because of inheritance, 'naivebayes' will have the same properties and
%   methods with superclass 'classifier'. So we only have to define unique
%   add-on functionalities in 'naivesbayes'.

   methods
       function p=docprob(self,item,cat)
           features=self.getfeatures(item);
           
           % Multiply the probabilities of all the features together
           p=1;
           for i=1:size(features,1)
               p=p*self.weightedprob(features{i},cat,@self.fprob);
           end
       end
       
       function p=prob(self,item,cat)
           catprob=self.catcount(cat)/self.totalcount();
           docprob=self.docprob(item,cat);
           p=docprob*catprob;
       end
   end
end 
