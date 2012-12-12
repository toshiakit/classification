function f=entryfeatures(entry)
% Split cell array of title, publisher and summary by non-alphabet 
% characters and return a cell array of lowercase words and UPPERCASE flag

% Extract the title words and annotate
titlewords=regexpi(lower(entry{1,1}), '\W', 'split');
len=cellfun('size',titlewords,2);
titlewords=titlewords(len<20);
titlewords=titlewords(len>2);
len=cellfun('size', regexp(titlewords,'[a-z]'),2);
titlewords=titlewords(cellfun('size',titlewords,2)==len);
titlewords=strcat('Title_', titlewords);

% Extract the summary words
summarywords=regexpi(entry{1,3}, '\W', 'split');
len=cellfun('size',summarywords,2);
summarywords=summarywords(len<20);
summarywords=summarywords(len>2);
len=cellfun('size', regexp(summarywords,'[a-zA-Z]'),2);
summarywords=summarywords(cellfun('size',summarywords,2)==len);

% Count uppercase words
len=cellfun('size', regexp(summarywords,'[A-Z]'),2);
uc=size(len(len>0),2);
% Convert to lowercase
summarywords=cellfun(@lower,summarywords,'UniformOutput',0);

% Get word pairs in summary as features
twowords=strcat(summarywords(1:end-1), '_', summarywords(2:end));

% Put together all the features into the output cell array
f=[titlewords'; summarywords'; twowords'];

% Add publisher as a whole
f{end+1,1}=['Publisher_' entry{1,2}];

% Add UPPERCASE as a feature if more than 30% of words are in uppercase
if uc/size(summarywords,2)>0.3
    f{end+1,1}='UPPERCASE';
end

 