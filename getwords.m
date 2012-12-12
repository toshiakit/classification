function words=getwords(doc)
% Split text string by non-alphabet characters and return a cell array of
% lowercase words
    
    words=regexpi(lower(doc), '\W', 'split');
    len=cellfun('size',words,2);
    % Words must be more then 2 letters long but less than 20 letters 
    words=words(len<20);
    words=words(len>2);
    % Make sure the word only have alphabet
    len=cellfun('size', regexp(words,'[a-z]'),2);
    words=words(cellfun('size',words,2)==len);
    words=words';