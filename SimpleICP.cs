using UnityEngine;
using System.Collections.Generic;
using System.Linq; // Required for .Sort() and .ToList()
using MathNet.Numerics.LinearAlgebra;

public static class SimpleICP
{
    // --- Helper struct to store a matched pair and its distance ---
    private struct Correspondence
    {
        public int sourceIndex;
        public int targetIndex;
        public float distanceSqr; // We use squared distance for performance
    }

    // --- Store a System.Random for RANSAC ---
    private static System.Random ransacRandom = new System.Random();

    private static Quaternion MatrixToQuaternion(Matrix<double> R_mathnet)
    {
        float m00 = (float)R_mathnet[0, 0]; float m01 = (float)R_mathnet[0, 1]; float m02 = (float)R_mathnet[0, 2];
        float m10 = (float)R_mathnet[1, 0]; float m11 = (float)R_mathnet[1, 1]; float m12 = (float)R_mathnet[1, 2];
        float m20 = (float)R_mathnet[2, 0]; float m21 = (float)R_mathnet[2, 1]; float m22 = (float)R_mathnet[2, 2];
        float trace = m00 + m11 + m22;
        float w, x, y, z;
        if (trace > 0.0f)
        {
            float s = Mathf.Sqrt(trace + 1.0f); w = s * 0.5f; s = 0.5f / s;
            x = (m21 - m12) * s; y = (m02 - m20) * s; z = (m10 - m01) * s;
        }
        else
        {
            if (m00 > m11 && m00 > m22)
            {
                float s = Mathf.Sqrt(1.0f + m00 - m11 - m22); x = s * 0.5f; s = 0.5f / s;
                y = (m10 + m01) * s; z = (m02 + m20) * s; w = (m21 - m12) * s;
            }
            else if (m11 > m22)
            {
                float s = Mathf.Sqrt(1.0f + m11 - m00 - m22); y = s * 0.5f; s = 0.5f / s;
                x = (m10 + m01) * s; z = (m21 + m12) * s; w = (m02 - m20) * s;
            }
            else
            {
                float s = Mathf.Sqrt(1.0f + m22 - m00 - m11); z = s * 0.5f; s = 0.5f / s;
                x = (m02 + m20) * s; y = (m21 + m12) * s; w = (m10 - m01) * s;
            }
        }
        return new Quaternion(x, y, z, w);
    }
    private static Matrix4x4 MatrixToUnityMatrix(Matrix<double> R_mathnet)
    {
        Matrix4x4 m = Matrix4x4.identity;
        m.m00 = (float)R_mathnet[0, 0]; m.m01 = (float)R_mathnet[0, 1]; m.m02 = (float)R_mathnet[0, 2];
        m.m10 = (float)R_mathnet[1, 0]; m.m11 = (float)R_mathnet[1, 1]; m.m12 = (float)R_mathnet[1, 2];
        m.m20 = (float)R_mathnet[2, 0]; m.m21 = (float)R_mathnet[2, 1]; m.m22 = (float)R_mathnet[2, 2];
        return m;
    }

    // Changed signature. Now transforms 'source' points into the 'destination'
    // list to avoid allocating a new list every time.
    private static void TransformPoints(List<Vector3> source, List<Vector3> destination, Matrix4x4 transform)
    {
        destination.Clear();
        // Ensure the list has enough capacity to avoid re-allocations
        if (destination.Capacity < source.Count)
        {
            destination.Capacity = source.Count;
        }

        foreach (var p in source)
        {
            destination.Add(transform.MultiplyPoint3x4(p));
        }
    }

    // --- SOLVER 1: NAIVE / SIMPLE ICP ---

    public static (Matrix4x4, float) Solve_Simple(List<Vector3> sourcePoints, List<Vector3> targetPoints,
        int maxIterations = 10, float tolerance = 0.001f, Matrix4x4? initialTransform = null)
    {
        // --- REFACTOR ---
        // Use the provided initialTransform, or default to identity.
        Matrix4x4 finalTransform = initialTransform ?? Matrix4x4.identity;

        // --- OPTIMIZATION ---
        // Create the buffer list ONCE outside the loop.
        List<Vector3> currentSource = new List<Vector3>(sourcePoints.Count);
        TransformPoints(sourcePoints, currentSource, finalTransform); // Initial population

        float lastError = float.MaxValue;

        for (int i = 0; i < maxIterations; i++)
        {
            // 'currentSource' is already transformed from the previous iteration
            List<Correspondence> allPairs = FindCorrespondences(currentSource, targetPoints);
            if (allPairs.Count == 0) break;

            List<Vector3> sourceForSolver = new List<Vector3>(allPairs.Count);
            List<Vector3> targetForSolver = new List<Vector3>(allPairs.Count);
            foreach (var pair in allPairs)
            {
                // We use 'currentSource' here, which is the *already transformed* point
                sourceForSolver.Add(currentSource[pair.sourceIndex]);
                targetForSolver.Add(targetPoints[pair.targetIndex]);
            }

            (Matrix4x4 deltaTransform, float error) = FindOptimalTransform(sourceForSolver, targetForSolver);

            // Apply the delta to our final transform
            finalTransform = deltaTransform * finalTransform;

            // --- OPTIMIZATION ---
            // Re-transform the *original* points with the *new* final transform
            // into our buffer.
            TransformPoints(sourcePoints, currentSource, finalTransform);

            if (Mathf.Abs(lastError - error) < tolerance)
            {
                Debug.Log($"[SimpleICP-Simple] Converged in {i + 1} iterations. Final RMS Error: {error:F5}");
                return (finalTransform, error);
            }
            lastError = error;
            if (i == maxIterations - 1)
            {
                Debug.LogWarning($"[SimpleICP-Simple] Hit max iterations ({maxIterations}). Final RMS Error: {error:F5}");
                return (finalTransform, error);
            }
        }
        return (finalTransform, lastError);
    }


    public static Matrix4x4 Solve_RANSAC(List<Vector3> sourcePoints, List<Vector3> targetPoints, int ransacIterations = 5000, float inlierThreshold = 0.04f, float distancePercentile = 0.5f)
    {
        float inlierThresholdSqr = inlierThreshold * inlierThreshold;

        // 1. Find all correspondences (Nearest Neighbor)
        List<Correspondence> allPairs = FindCorrespondences(sourcePoints, targetPoints);
        if (allPairs.Count < 3)
        {
            Debug.LogWarning("[RANSAC] Not enough pairs to solve.");
            return Matrix4x4.identity;
        }

        // 2. Pre-filter the pairs
        allPairs.Sort((a, b) => a.distanceSqr.CompareTo(b.distanceSqr));
        int pairsToKeep = (int)(allPairs.Count * distancePercentile);
        if (pairsToKeep < 3) pairsToKeep = allPairs.Count;

        // 'trimmedPairs' is now a smaller, much higher-quality list
        List<Correspondence> trimmedPairs = allPairs.GetRange(0, pairsToKeep);

        Debug.Log($"[ICP-RANSAC] Pre-filter: Kept {trimmedPairs.Count} / {allPairs.Count} pairs.");

        Matrix4x4 bestTransform = Matrix4x4.identity;
        int bestInlierCount = -1;

        // --- OPTIMIZATION ---
        // Pre-allocate the lists ONCE with capacity.
        List<Correspondence> bestInlierSet = new List<Correspondence>(trimmedPairs.Count);
        List<Correspondence> currentInlierSet = new List<Correspondence>(trimmedPairs.Count);

        // We pre-allocate these sample lists once to avoid generating garbage
        List<Vector3> sourceSample = new List<Vector3>(3);
        List<Vector3> targetSample = new List<Vector3>(3);

        // --- 3. RANSAC Loop ---
        for (int i = 0; i < ransacIterations; i++)
        {
            sourceSample.Clear();
            targetSample.Clear();

            HashSet<int> usedIndices = new HashSet<int>();
            while (usedIndices.Count < 3) { usedIndices.Add(ransacRandom.Next(0, trimmedPairs.Count)); }

            foreach (int index in usedIndices)
            {
                sourceSample.Add(sourcePoints[trimmedPairs[index].sourceIndex]);
                targetSample.Add(targetPoints[trimmedPairs[index].targetIndex]);
            }

            (Matrix4x4 candidateTransform, float error) = FindOptimalTransform(sourceSample, targetSample);

            int currentInlierCount = 0;

            // --- OPTIMIZATION ---
            // Clear the list, don't create a new one.
            currentInlierSet.Clear();

            for (int j = 0; j < trimmedPairs.Count; j++)
            {
                var pair = trimmedPairs[j];
                Vector3 transformedSource = candidateTransform.MultiplyPoint3x4(sourcePoints[pair.sourceIndex]);
                float distSqr = (transformedSource - targetPoints[pair.targetIndex]).sqrMagnitude;

                if (distSqr < inlierThresholdSqr)
                {
                    currentInlierCount++;
                    currentInlierSet.Add(pair);
                }
            }

            // d. Check if this is the best model so far
            if (currentInlierCount > bestInlierCount)
            {
                bestInlierCount = currentInlierCount;
                bestTransform = candidateTransform;

                // --- OPTIMIZATION ---
                // Swap list references. This is a zero-allocation, O(1) operation.
                var tempList = bestInlierSet;
                bestInlierSet = currentInlierSet;
                currentInlierSet = tempList;
            }
        }

        Debug.Log($"[RANSAC] Best model found with {bestInlierCount} / {trimmedPairs.Count} inliers.");

        // --- 4. Final Polish ---
        // We use 'bestInlierSet' which now contains the inliers from the best model.
        if (bestInlierCount > 3)
        {
            List<Vector3> sourceFinal = new List<Vector3>(bestInlierCount);
            List<Vector3> targetFinal = new List<Vector3>(bestInlierCount);
            foreach (var pair in bestInlierSet)
            {
                // We use the *original* points from the inlier set
                sourceFinal.Add(sourcePoints[pair.sourceIndex]);
                targetFinal.Add(targetPoints[pair.targetIndex]);
            }

            (Matrix4x4 polishTransform, _) = FindOptimalTransform(sourceFinal, targetFinal);

            // 'polishTransform' is the hyper-accurate transform. We return it.
            return polishTransform;
        }

        // If we failed, just return the "best guess" we found
        return bestTransform;
    }

    /// <summary>
    /// Solves for the optimal transform using a robust RANSAC-then-Iterative method.
    /// This provides RANSAC's robustness against outliers and SimpleICP's precision and stability.
    /// This is the recommended function to call from your PacketReceiver.
    /// </summary>
    /// <param name="ransacIterations">Iterations for the initial RANSAC guess.</param>
    /// <param name="simpleIterations">Iterations for the SimpleICP refinement.</param>
    public static (Matrix4x4, float) Solve_Robust(List<Vector3> sourcePoints, List<Vector3> targetPoints,
        int ransacIterations = 5000, float inlierThreshold = 0.04f, float distancePercentile = 0.75f,
        int simpleIterations = 10, float simpleTolerance = 0.001f)
    {
        Debug.Log($"[ICP-Robust] Running {ransacIterations} iterations of RANSAC");
        // 1. Get the robust RANSAC guess
        Matrix4x4 ransacTransform = Solve_RANSAC(
            sourcePoints,
            targetPoints,
            ransacIterations,
            inlierThreshold,
            distancePercentile
        );

        if (ransacTransform == Matrix4x4.identity)
        {
            Debug.LogWarning("[ICP-Robust] RANSAC step failed or returned identity. Result may be inaccurate.");
            // We'll still let Solve_Simple try to refine from identity.
        }

        // 2. Use the RANSAC result to seed the precise iterative solver
        return Solve_Simple(
            sourcePoints,
            targetPoints,
            simpleIterations,
            simpleTolerance,
            ransacTransform // <-- Here is the seed
        );


    }


    /// <summary>
    /// Finds all nearest-neighbor correspondences. (Helper)
    /// </summary>
    private static List<Correspondence> FindCorrespondences(List<Vector3> source, List<Vector3> target)
    {
        // --- NOTE ---
        // This O(N*M) loop is the main bottleneck for large clouds.
        // It can be optimized to O((N+M)logM) by building a K-d tree
        // on the 'target' points and querying it for each 'source' point.
        // For now, this is correct, just potentially slow.

        var correspondingPairs = new List<Correspondence>(source.Count);
        for (int i = 0; i < source.Count; i++)
        {
            float minSqrDist = float.MaxValue;
            int bestMatchIndex = 0;
            for (int j = 0; j < target.Count; j++)
            {
                float sqrDist = (source[i] - target[j]).sqrMagnitude;
                if (sqrDist < minSqrDist)
                {
                    minSqrDist = sqrDist;
                    bestMatchIndex = j;
                }
            }
            correspondingPairs.Add(new Correspondence
            {
                sourceIndex = i,
                targetIndex = bestMatchIndex,
                distanceSqr = minSqrDist
            });
        }
        return correspondingPairs;
    }

    /// <summary>
    /// The core SVD solver. Takes two *pre-matched* lists. (Helper)
    /// </summary>
    private static (Matrix4x4, float) FindOptimalTransform(List<Vector3> source, List<Vector3> target)
    {
        int N = source.Count;
        if (N < 3) return (Matrix4x4.identity, 0); // Need at least 3 points

        // Find Centroids (just the average)
        Vector3 centroidSource = Vector3.zero;
        Vector3 centroidTarget = Vector3.zero;
        for (int i = 0; i < N; i++)
        {
            centroidSource += source[i];
            centroidTarget += target[i];
        }

        centroidSource /= N;
        centroidTarget /= N;

        // 2. Build Covariance Matrix H
        var H = Matrix<double>.Build.Dense(3, 3, 0.0);
        for (int i = 0; i < N; i++)
        {
            var p_prime = source[i] - centroidSource;
            var q_prime = target[i] - centroidTarget;
            H[0, 0] += p_prime.x * q_prime.x; H[0, 1] += p_prime.x * q_prime.y; H[0, 2] += p_prime.x * q_prime.z;
            H[1, 0] += p_prime.y * q_prime.x; H[1, 1] += p_prime.y * q_prime.y; H[1, 2] += p_prime.y * q_prime.z;
            H[2, 0] += p_prime.z * q_prime.x; H[2, 1] += p_prime.z * q_prime.y; H[2, 2] += p_prime.z * q_prime.z;
        }

        // 3. Perform SVD (H = U * S * V^T)
        var svd = H.Svd(true);
        Matrix<double> U = svd.U;
        Matrix<double> V = svd.VT.Transpose();

        // 4. Calculate Rotation R = V * U.Transpose()
        Matrix<double> R_mathnet = V.Multiply(U.Transpose());

        // 5. Handle Reflection Case
        if (R_mathnet.Determinant() < 0)
        {
            var V_copy = V.Clone(); // Don't modify the original V
            V_copy.SetColumn(2, V_copy.Column(2) * -1.0);
            R_mathnet = V_copy.Multiply(U.Transpose());
        }

        // 6. Convert Math.NET Matrix to Unity Quaternion and Matrix4x4
        Quaternion R_quat = MatrixToQuaternion(R_mathnet);
        Matrix4x4 R_unity = MatrixToUnityMatrix(R_mathnet);

        // 7. Calculate Translation t = centroid_q - (R * centroid_p) 
        Vector3 rotatedCentroidSource = R_unity.MultiplyPoint3x4(centroidSource);
        Vector3 t = centroidTarget - rotatedCentroidSource;

        // 8. Combine into final Unity Matrix
        Matrix4x4 transform = Matrix4x4.TRS(t, R_quat, Vector3.one);

        // 9. Calculate RMS error
        float error = 0;
        for (int i = 0; i < N; i++)
        {
            error += (transform.MultiplyPoint3x4(source[i]) - target[i]).sqrMagnitude;
        }
        return (transform, error / N);
    }


    /// <summary>
    /// Finds a robust 4-DoF (Yaw+Position) transform guess using RANSAC.
    /// This is for "leveled" point clouds (where Pitch/Roll are already solved).
    /// </summary>
    /// <returns>A (Transform, RMS_Error) tuple from the final polished result.</returns>
    public static (Matrix4x4, float) Solve_RANSAC_4DoF(List<Vector3> sourcePoints, List<Vector3> targetPoints,
        int ransacIterations = 3000, float inlierThreshold = 0.04f, float distancePercentile = 0.8f)
    {
        float inlierThresholdSqr = inlierThreshold * inlierThreshold;

        Debug.Log("ICP - Running Solve_RANSAC_4DOF");
        // 1. Find all correspondences
        List<Correspondence> allPairs = FindCorrespondences(sourcePoints, targetPoints);

        // --- MODIFIED: Need at least 2 points for 2D/4-DoF ---
        if (allPairs.Count < 2)
        {
            Debug.LogWarning("[RANSAC-4DoF] Not enough pairs to solve.");
            return (Matrix4x4.identity, 0);
        }

        // 2. Pre-filter the pairs
        allPairs.Sort((a, b) => a.distanceSqr.CompareTo(b.distanceSqr));
        int pairsToKeep = (int)(allPairs.Count * distancePercentile);
        if (pairsToKeep < 2) pairsToKeep = allPairs.Count; // --- MODIFIED ---

        List<Correspondence> trimmedPairs = allPairs.GetRange(0, pairsToKeep);
        Debug.Log($"[RANSAC-4DoF] Pre-filter: Kept {trimmedPairs.Count} / {allPairs.Count} pairs.");

        Matrix4x4 bestTransform = Matrix4x4.identity;
        int bestInlierCount = -1;
        List<Correspondence> bestInlierSet = new List<Correspondence>(trimmedPairs.Count);
        List<Correspondence> currentInlierSet = new List<Correspondence>(trimmedPairs.Count);

        // --- MODIFIED: Need 2 points ---
        List<Vector3> sourceSample = new List<Vector3>(2);
        List<Vector3> targetSample = new List<Vector3>(2);

        // --- 3. RANSAC Loop ---
        for (int i = 0; i < ransacIterations; i++)
        {
            sourceSample.Clear();
            targetSample.Clear();

            // --- MODIFIED: Sample 2 points ---
            HashSet<int> usedIndices = new HashSet<int>();
            while (usedIndices.Count < 2) { usedIndices.Add(ransacRandom.Next(0, trimmedPairs.Count)); }

            foreach (int index in usedIndices)
            {
                sourceSample.Add(sourcePoints[trimmedPairs[index].sourceIndex]);
                targetSample.Add(targetPoints[trimmedPairs[index].targetIndex]);
            }

            // --- MODIFIED: Call 4-DoF helper ---
            (Matrix4x4 candidateTransform, float error) = FindOptimalTransform_4DoF(sourceSample, targetSample);

            int currentInlierCount = 0;
            currentInlierSet.Clear();

            // Inlier check is still a 3D distance check, which is correct
            for (int j = 0; j < trimmedPairs.Count; j++)
            {
                var pair = trimmedPairs[j];
                Vector3 transformedSource = candidateTransform.MultiplyPoint3x4(sourcePoints[pair.sourceIndex]);
                float distSqr = (transformedSource - targetPoints[pair.targetIndex]).sqrMagnitude;

                if (distSqr < inlierThresholdSqr)
                {
                    currentInlierCount++;
                    currentInlierSet.Add(pair);
                }
            }

            if (currentInlierCount > bestInlierCount)
            {
                bestInlierCount = currentInlierCount;
                bestTransform = candidateTransform;
                // Swap lists
                var tempList = bestInlierSet;
                bestInlierSet = currentInlierSet;
                currentInlierSet = tempList;
            }
        }

        Debug.Log($"[RANSAC-4DoF] Best model found with {bestInlierCount} / {trimmedPairs.Count} inliers.");

        // --- 4. Final Polish ---
        // --- MODIFIED: Need > 2 inliers ---
        if (bestInlierCount > 2)
        {
            List<Vector3> sourceFinal = new List<Vector3>(bestInlierCount);
            List<Vector3> targetFinal = new List<Vector3>(bestInlierCount);
            foreach (var pair in bestInlierSet)
            {
                sourceFinal.Add(sourcePoints[pair.sourceIndex]);
                targetFinal.Add(targetPoints[pair.targetIndex]);
            }

            // --- MODIFIED: Call 4-DoF helper ---
            // Polish the result using all inliers
            (Matrix4x4 polishTransform, float polishError) = FindOptimalTransform_4DoF(sourceFinal, targetFinal);

            return (polishTransform, polishError);
        }

        // Failed to find a good model, return the best guess (which is likely identity)
        return (bestTransform, float.MaxValue);
    }
    


    private static (Matrix4x4, float) FindOptimalTransform_4DoF(List<Vector3> source, List<Vector3> target)
    {
        int N = source.Count;
        if (N < 2) return (Matrix4x4.identity, 0);

        Vector3 centroidSource = Vector3.zero;
        Vector3 centroidTarget = Vector3.zero;
        for (int i = 0; i < N; i++)
        {
            centroidSource += source[i];
            centroidTarget += target[i];
        }
        centroidSource /= N;
        centroidTarget /= N;

        // Build 2D Covariance (X, Z)
        var H_2D = Matrix<double>.Build.Dense(2, 2, 0.0);
        for (int i = 0; i < N; i++)
        {
            var p = source[i] - centroidSource;
            var q = target[i] - centroidTarget;

            H_2D[0, 0] += p.x * q.x; H_2D[0, 1] += p.x * q.z;
            H_2D[1, 0] += p.z * q.x; H_2D[1, 1] += p.z * q.z;
        }

        var svd = H_2D.Svd(true);
        Matrix<double> U = svd.U;
        Matrix<double> V = svd.VT.Transpose();

        // --- REFLECTION CHECK ---
        Matrix<double> R_test = V.Multiply(U.Transpose());
        double det = (R_test[0, 0] * R_test[1, 1]) - (R_test[0, 1] * R_test[1, 0]);

        if (det < 0)
        {
            var V_copy = V.Clone();
            V_copy.SetColumn(1, V_copy.Column(1) * -1.0);
            R_test = V_copy.Multiply(U.Transpose());
        }

        float cos_yaw = (float)R_test[0, 0];
        float sin_yaw = (float)R_test[1, 0];
        float yaw_rad = Mathf.Atan2(sin_yaw, cos_yaw);

        Quaternion R_quat_3D = Quaternion.Euler(0, yaw_rad * Mathf.Rad2Deg, 0);
        Vector3 t = centroidTarget - (R_quat_3D * centroidSource);

        Matrix4x4 transform = Matrix4x4.TRS(t, R_quat_3D, Vector3.one);

        float error = 0;
        for (int i = 0; i < N; i++)
        {
            error += (transform.MultiplyPoint3x4(source[i]) - target[i]).sqrMagnitude;
        }
        return (transform, error / N);
    }

    // --- 2. The Iterative Solver (With Outlier Rejection) ---
    public static (Matrix4x4, float) Solve_4DoF(List<Vector3> sourcePoints, List<Vector3> targetPoints,
        int maxIterations = 10, float tolerance = 0.001f, Matrix4x4? initialTransform = null)
    {
        Matrix4x4 finalTransform = initialTransform ?? Matrix4x4.identity;
        List<Vector3> currentSource = new List<Vector3>(sourcePoints.Count);
        TransformPoints(sourcePoints, currentSource, finalTransform);

        // --- SETTING: Rejection Threshold ---
        // In a small 1m area, if a point's closest neighbor is > 10cm away, 
        // it is likely noise or non-overlapping data. Ignore it.
        float rejectionDistSqr = 0.1f * 0.1f;

        float lastError = float.MaxValue;

        for (int i = 0; i < maxIterations; i++)
        {
            List<Correspondence> allPairs = FindCorrespondences(currentSource, targetPoints);
            if (allPairs.Count == 0) break;

            List<Vector3> srcSolver = new List<Vector3>();
            List<Vector3> tgtSolver = new List<Vector3>();

            // --- FILTERING STEP ---
            foreach (var pair in allPairs)
            {
                // Only accept "good" matches
                if (pair.distanceSqr < rejectionDistSqr)
                {
                    srcSolver.Add(currentSource[pair.sourceIndex]);
                    tgtSolver.Add(targetPoints[pair.targetIndex]);
                }
            }

            // Safety: If we filter too aggressively and lose all points, 
            // stop and return the best we have so far.
            if (srcSolver.Count < 5)
            {
                // Optional: If you want to be aggressive, you could increase threshold here.
                // For now, we just exit to avoid instability.
                break;
            }

            // Run solver only on the GOOD points
            (Matrix4x4 delta, float error) = FindOptimalTransform_4DoF(srcSolver, tgtSolver);

            finalTransform = delta * finalTransform;
            TransformPoints(sourcePoints, currentSource, finalTransform);

            if (Mathf.Abs(lastError - error) < tolerance) return (finalTransform, error);
            lastError = error;
        }
        return (finalTransform, lastError);
    }

    public static (Matrix4x4, float) Solve_4DoF_Anneal(List<Vector3> sourcePoints, List<Vector3> targetPoints,
        int maxIterations = 15, float tolerance = 0.001f, Matrix4x4? initialTransform = null)
    {
        Matrix4x4 finalTransform = initialTransform ?? Matrix4x4.identity;
        List<Vector3> currentSource = new List<Vector3>(sourcePoints.Count);
        TransformPoints(sourcePoints, currentSource, finalTransform);

        // --- ANNEALING CONFIGURATION ---
        // Start by accepting matches up to 20cm away (helps "grab" the cloud in larger areas)
        float currentMaxDist = 0.20f;
        // Don't ever go tighter than 1cm (avoids rejecting valid noise)
        float minMaxDist = 0.01f;
        // How fast to tighten? (0.7 means shrink by 30% each step)
        float annealFactor = 0.85f;

        float lastError = float.MaxValue;

        for (int i = 0; i < maxIterations; i++)
        {
            // Calculate squared threshold for this specific iteration
            float rejectionDistSqr = currentMaxDist * currentMaxDist;

            List<Correspondence> allPairs = FindCorrespondences(currentSource, targetPoints);
            if (allPairs.Count == 0) break;

            List<Vector3> srcSolver = new List<Vector3>();
            List<Vector3> tgtSolver = new List<Vector3>();

            // Filter based on the CURRENT (annealed) threshold
            foreach (var pair in allPairs)
            {
                if (pair.distanceSqr < rejectionDistSqr)
                {
                    srcSolver.Add(currentSource[pair.sourceIndex]);
                    tgtSolver.Add(targetPoints[pair.targetIndex]);
                }
            }

            // If we lost too many points, don't run the solver this step, 
            // but maybe next step (with different threshold) would be weird. 
            // Usually better to just break or return best.
            if (srcSolver.Count < 5)
            {
                // If we are at the very start and fail, return identity/initial
                if (i == 0) return (finalTransform, lastError);
                break;
            }

            (Matrix4x4 delta, float error) = FindOptimalTransform_4DoF(srcSolver, tgtSolver);

            finalTransform = delta * finalTransform;
            TransformPoints(sourcePoints, currentSource, finalTransform);

            // --- ANNEAL STEP ---
            // Tighten the search radius for the next loop
            currentMaxDist *= annealFactor;
            if (currentMaxDist < minMaxDist) currentMaxDist = minMaxDist;

            if (Mathf.Abs(lastError - error) < tolerance) return (finalTransform, error);
            lastError = error;
        }
        return (finalTransform, lastError);
    }
    
    // --- NEW: "Blind" Global RANSAC (Does not assume initial alignment) ---
    public static (Matrix4x4, float) Solve_RANSAC_4DoF_stick(List<Vector3> sourcePoints, List<Vector3> targetPoints,
        int ransacIterations = 3000, float inlierThreshold = 0.05f, float distancePercentile = 1.0f) // Use all points
    {
        int N = sourcePoints.Count;
        int M = targetPoints.Count;
        
        // If clouds are too small, we can't match geometry
        if (N < 2 || M < 2) return (Matrix4x4.identity, float.MaxValue);

        float inlierThresholdSqr = inlierThreshold * inlierThreshold;
        Matrix4x4 bestTransform = Matrix4x4.identity;
        int bestInlierCount = -1;

        // Optimization: We need to find pairs in Target that match distance L.
        // Since M is small (~300), we can just random sample, or build a lookup if needed.
        // For 300 points, random sampling is usually sufficient and faster than building a structure.

        // Reusable lists for the solver
        List<Vector3> srcSample = new List<Vector3>(2);
        List<Vector3> tgtSample = new List<Vector3>(2);

        // Tolerance for "Distance Matching" (e.g., 1cm difference in length is allowed)
        float distTolerance = 0.01f; 

        for (int i = 0; i < ransacIterations; i++)
        {
            // 1. Pick a random pair in Source (The "Stick")
            int s1 = ransacRandom.Next(N);
            int s2 = ransacRandom.Next(N);
            
            // Ensure they aren't the same point and have some length
            while (s1 == s2) s2 = ransacRandom.Next(N);
            
            float sourceDist = Vector3.Distance(sourcePoints[s1], sourcePoints[s2]);
            
            // If the stick is too short, it gives bad rotation accuracy. Skip it.
            if (sourceDist < 0.2f) continue; 

            // 2. Find a matching "Stick" in Target
            // We try up to 20 times to find a pair in Target with similar length
            int t1 = -1, t2 = -1;
            bool foundMatch = false;

            for (int k = 0; k < 20; k++)
            {
                int tryT1 = ransacRandom.Next(M);
                int tryT2 = ransacRandom.Next(M);
                if (tryT1 == tryT2) continue;

                float targetDist = Vector3.Distance(targetPoints[tryT1], targetPoints[tryT2]);

                // Does the length match?
                if (Mathf.Abs(sourceDist - targetDist) < distTolerance)
                {
                    t1 = tryT1;
                    t2 = tryT2;
                    foundMatch = true;
                    break;
                }
            }

            if (!foundMatch) continue;

            // 3. Compute Transform (Align Source Stick to Target Stick)
            srcSample.Clear(); tgtSample.Clear();
            srcSample.Add(sourcePoints[s1]); srcSample.Add(sourcePoints[s2]);
            tgtSample.Add(targetPoints[t1]); tgtSample.Add(targetPoints[t2]);

            (Matrix4x4 candidate, float _) = FindOptimalTransform_4DoF(srcSample, tgtSample);

            // 4. Score it (Count Inliers)
            int currentInlierCount = 0;
            
            // Optimization: For global registration, we can verify a subset first
            // but with only 300 points, checking all is fine.
            for (int j = 0; j < N; j++)
            {
                Vector3 pTrans = candidate.MultiplyPoint3x4(sourcePoints[j]);
                
                // We still need nearest neighbor for scoring, 
                // BUT we only check if there is a match CLOSE by.
                // We don't iterate the whole list if we can avoid it, 
                // but for N=300, O(N*M) is acceptable here (300*300 = 90k ops * 3000 iters is too slow).
                
                // Faster Scoring: Just check a few random points? 
                // No, we need accuracy.
                // Let's assume we check against the closest point.
                
                // To make this fast enough for Update loop:
                // We only check if this transform maps pTrans to *somewhere* in Target bounds?
                // No, let's do a quick "Broad Phase" rejection?
                // Actually, let's just check 10 random source points first. 
                // If they don't match, discard the model.
                
                // -- Quick Rejection --
                // (This prevents running the full O(N*M) loop for bad models)
            }
            
            // FULL SCORE (Simplified for performance)
            // We calculate error for 20 random points. If good, we check all.
            int score = 0;
            int sampleCount = 20; // Check 20 points
            for(int check = 0; check < sampleCount; check++) {
                 int idx = ransacRandom.Next(N);
                 Vector3 pT = candidate.MultiplyPoint3x4(sourcePoints[idx]);
                 // Find closest in Target (Brute force for now, optimize if slow)
                 float minDist = float.MaxValue;
                 for(int m=0; m<M; m++) {
                     float d = (pT - targetPoints[m]).sqrMagnitude;
                     if(d < minDist) minDist = d;
                 }
                 if(minDist < inlierThresholdSqr) score++;
            }

            // If our random sample looked promising (e.g. > 50% fit), do a full check
            if (score > (sampleCount / 2)) 
            {
                currentInlierCount = 0;
                for (int j = 0; j < N; j++)
                {
                    Vector3 pT = candidate.MultiplyPoint3x4(sourcePoints[j]);
                    float minDist = float.MaxValue;
                    // Find closest
                    for(int m=0; m<M; m++) {
                        float d = (pT - targetPoints[m]).sqrMagnitude;
                        if(d < minDist) minDist = d;
                        if (d < inlierThresholdSqr) break; // Early exit optimization
                    }
                    if (minDist < inlierThresholdSqr) currentInlierCount++;
                }

                if (currentInlierCount > bestInlierCount)
                {
                    bestInlierCount = currentInlierCount;
                    bestTransform = candidate;
                }
            }
        }
        
        // If we found a valid model (e.g. > 20% overlap)
        if (bestInlierCount > (N * 0.2f)) 
        {
            return (bestTransform, 0.0f); // Return the rough guess
        }
        
        return (Matrix4x4.identity, float.MaxValue);
    }
}