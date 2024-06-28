function [X,bandwidth] = smile(N)
    eye_pts = ceil(sqrt(N));
    mouth_pts = ceil(N/10);
    face_pts = N - 2*eye_pts - mouth_pts;

    X = zeros(N,2);
    idx = 1;
    for eye_center = [-4,4]
        eye_pts_gen = 0;
        while eye_pts_gen < eye_pts
            x = 2*rand()-1;
            y = 2*rand()-1;
            if x^2+y^2 <= 1
                X(idx,1) = x + eye_center;
                X(idx,2) = y + 4;
                idx = idx+1;
                eye_pts_gen = eye_pts_gen+1;
            end
        end
    end
    
    for x = linspace(-5,5,mouth_pts)
        X(idx,1) = x;
        X(idx,2) = x^2/16 - 5;
        idx = idx+1;
    end

    ts = linspace(0,2*pi,face_pts+1);
    ts = ts(1:face_pts);
    for t = ts
        X(idx,1) = 10 * cos(t);
        X(idx,2) = 10 * sin(t);
        idx = idx+1;
    end

    bandwidth = 1;

end