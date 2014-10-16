hidden_nodes = zeros(size(train_x, 1), length(dbn.rbm{end}.b));
for loop = 1 : size(train_x, 1)
    hidden_nodes(loop, :) = rbmup(dbn.rbm{2}, rbmup(dbn.rbm{1}, train_x(loop, :)));
    x = rbmdown(dbn.rbm{1}, rbmdown(dbn.rbm{2}, hidden_nodes(loop, :)));
    
%     subplot(1, 2, 1);
%     imshow(reshape(train_x(loop, :), sqrt(length(x)), [])')
%     title(num2str(find(train_y(loop, :))-1))
%     
%     subplot(1, 2, 2);
%     imshow(reshape(x, sqrt(length(x)), [])')
%     title(num2str(find(train_y(loop, :))-1))
%     pause
end

%%
figure;
for featLoop = 1 : 100
    clf;
    for digLoop = 1 : 10
        subplot(2, 5, digLoop);
        [counts,vals] = hist(hidden_nodes(train_y(:, digLoop) == 1, featLoop), 20);
        bar(vals, counts/sum(counts))
        set(gca, 'YLim', [0 1])
        title(num2str(digLoop - 1));
    end
    pause
end

%%

numPoints = 1000;
dists = squareform(pdist(hidden_nodes(1:numPoints, :)));
scaledDists = mdscale(dists, 2);
cMap = lines(10);
figure;
hold on;
for digLoop = 1 : 10
    plot(scaledDists(train_y(1:numPoints, digLoop) == 1, 1), scaledDists(train_y(1:numPoints, digLoop) == 1, 2), 'LineStyle', 'o', 'Color', cMap(digLoop, :))
end