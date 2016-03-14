function X = FeatureScaling(phi,m);

% :::::::: Mean Normalization 

features = columns(phi);
S = max(phi)-min(phi);
U = mean(phi); %promedio

X = ones(m,1);

    for i=1:features

        X =[ X ( phi( : , i ) - U( i ) ) / S(i) ];

    end


end