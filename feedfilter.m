function feedfilter(classifier,url)
% Accept 'classifier' object and Google Blog Search URL or equivalent and
% display parsed text for interactive training.

    if nargin <2
        url='python_search.xml';
    end

    % Parse XML into a cell array 'items'
    % This doesn't work with all RSS feeds.
    f=xmlread(url);
    allitems=f.getElementsByTagName('item');
    entries=cell(allitems.getLength,3);
    for i=0:size(entries,1)-1
        thisitem=allitems.item(i);
        childNode=thisitem.getFirstChild;

        while ~isempty(childNode)
            if childNode.getNodeType==childNode.ELEMENT_NODE
                childText=char(childNode.getFirstChild.getData);
                switch char(childNode.getTagName)
                    case 'title';
                        entries{i+1,1}=childText;
                    case 'dc:publisher';
                        entries{i+1,2}=childText;
                    case 'description';
                        entries{i+1,3}=childText;
                end
            end
            childNode=childNode.getNextSibling;
        end
    end

    % Start the interactive session
    disp(' ')
    disp(sprintf('Start interactive training: %d entries', size(entries,1)))
    for i=1:size(entries,1)
        disp(' ')
        disp('-----')
        % Print the content of the entry
        disp(sprintf('Entry # %d',i))
        disp(sprintf('Title: %s', entries{i,1}))
        disp(sprintf('Publisher: %s', entries{i,2}))
        disp(' ')
        disp(entries{i,3})

        if strcmp(char(classifier.getfeatures),'entryfeatures')
            fulltext=entries(i,:);
        else
        % Combine all the text to create one item for the classifier
            fulltext=sprintf('%s\n%s\n%s', entries{i,1},entries{i,2},entries{i,3});
        end
        % Print the best guess at the current category
        guess=classifier.classify(fulltext);
        disp(['Guess: ' guess]);

        % Ask the user to specify the correct category
        if ~isempty(guess)
            % Ask to confirm the quess or enter new category
            prompt=sprintf('Confirm category? [%s] or enter new category: ',guess);
        else
            % Ask to enter new category
            prompt='Etner category: ';
        end
        user_entry = input(prompt,'s');
        % User data exits
        if ~isempty(user_entry)
            % If it is not 'end' use the user data for training
            if ~strcmp(user_entry,'end')
            classifier.train(fulltext,user_entry);
            else % If it is 'end', end the interactive session
                break;
            end
        else % If user data doesn't exit, user accepted the guess
            if ~isempty(guess)
                classifier.train(fulltext,guess);
            end
        end
    end
    disp(' ')
    disp('End of interactive training')
    disp('Thank you for your help.')
    disp(' ')


