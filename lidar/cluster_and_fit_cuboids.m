% Original MS thesis code (2021), Kamal Mehta. See README.md.
pc = pcread('1.pcd');
pcwonoise = pcdenoise(pc,'NumNeighbors',25);

[labels, numClusters]= pcsegdist(pcwonoise,0.05);
pcshow(pcwonoise)
for i = 1:numClusters
    idx = find(labels == i);
    model = pcfitcuboid(pcwonoise,idx);
    plot(model)
end

%%
pcshow(pcwonoise.Location,labels)
colormap(hsv(numClusters))

[model,inlierIndices,outlierIndices,meanError] = pcfitplane(pcwonoise,0.5,[0 1 0]);
plane = select(pcwonoise,inlierIndices);
remainPtCloud = select(pcwonoise,outlierIndices);
pcshow(plane)