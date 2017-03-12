function keyboard_input()

    key = get(gcbf, 'CurrentCharacter');
    
    if ~isempty(key)
        switch (key)
            case 'a'
                fprintf('...a...\n')
            case 'b'
                fprintf('...b...\n')
        end
    end

end