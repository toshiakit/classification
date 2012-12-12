function sampletrain(obj)
% provide sample training data into a classifier object
    train(obj,'Nobody owns the water.','good');
    train(obj,'the quick rabbit jumps fences','good');
    train(obj,'buy pharmaceuticals now','bad');
    train(obj,'make quick money at the online casino','bad');
    train(obj,'the quick brown fox jumps','good');
end
    