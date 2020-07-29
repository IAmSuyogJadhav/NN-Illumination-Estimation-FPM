function parameters = readMyParameters(parameterPath)

fid = fopen(parameterPath);
cac = textscan( fid, '%s%f' );
fclose( fid );

for jj = 1 : length(cac{1})
   parameters.( cac{1}{jj} ) = cac{2}(jj);   % dynamic names in a structure
   assignin( 'caller', cac{1}{jj}, cac{2}(jj) ); % static names in the workspace
end

end