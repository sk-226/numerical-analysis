function lbl = prec_config2str(config)
    %PREC_CONFIG2STR  convert a preconditioner configuration to a string
    %
    %   Example:
    %     config = struct('type','ssor','omega',1.0);
    %     lbl = prec_config2str(config)   % --> "ssor_omega1.0"
    %

    switch config.type
        case "none"
            lbl = "none";

        case "diag"
            lbl = "diag";

        case "ssor"
            lbl = "ssor_omega" + num2str(config.omega);

        case "ic"
            % Construct "ic_<ictype>_droptol<droptol>"
            if isfield(config, "ictype")
                ict = config.ictype;
            else
                ict = "nofill";
            end

            if isfield(config, "droptol")
                dt = config.droptol;
            else
                dt = 0;
            end

            lbl = "ic_" + string(ict) + "_droptol" + num2str(dt);

        otherwise
            % unknown type
            lbl = string(config.type);
    end

end
