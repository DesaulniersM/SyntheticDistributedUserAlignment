using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Collections.Concurrent;
using System.Collections.Generic;

using System.Diagnostics;

public class PacketReceiver : MonoBehaviour
{

    // TODO Delete this Stopwatch
    Stopwatch stopwatch = new Stopwatch();

    // --- Helper struct for the queue ---
    private struct ReceiveWorkItem
    {
        public byte[] Buffer; // The buffer containing the data
        public int Length;    // The length of the received data
        public double ReceiveTimestamp;
    }

    // --- Fields ---
    private Socket _receiveSocket; // --- REFACTOR: Switched from UdpClient to Socket
    private Thread receiveThread;
    private volatile bool running;

    // --- REFACTOR: New buffer pool and work queue
    private ConcurrentBag<byte[]> bufferPool = new();
    private ConcurrentQueue<ReceiveWorkItem> queue = new();

    // --- REFACTOR: Stores deserialized points, not the whole packet/buffer
    private Dictionary<int, List<Vector3>> _pendingPointClouds = new Dictionary<int, List<Vector3>>();

    [Header("Component References")]
    public ARCloudAnchorManager arCloudAnchorManager;
    public PointCloudProvider localPointCloudProvider;

    [Header("Prefabs & Visualizers")]
    public GameObject replicatedArObjectPrefab;
    private GameObject activeReplicatedObject;
    public PointCloudVisualizer_CPU hostVisualizer_CPU;

    [Header("ICP Solver Settings")]
    public bool useRansacSolver = true;

    [SerializeField]
    [Tooltip("After frustum culling, only use local points closer than this (meters).")]
    private float localMaxDistance = 1.3f; // 10-meter cutoff
    private float localMaxDistanceSqr;

    private Camera receiverCamera;

    // --- REFACTOR: Pre-allocated lists for zero-alloc processing ---
    private List<Vector3> _hostPointsWorld = new List<Vector3>();
    private List<Vector3> _relativePointsBuffer = new List<Vector3>();
    private List<Vector3> _localPointsBuffer = new List<Vector3>();

    private List<Vector3> _filteredLocalPoints = new List<Vector3>();

    [SerializeField]
    [Tooltip("If RMS error (m) exceeds this, re-run RANSAC to find a new pose.")]
    private float icpErrorResetThreshold = 0.05f; // 5cm
    private Matrix4x4? _lastKnownHostTransform = null; // Our state


    // // --- Add these new fields ---
    [Header("Sensor Fusion")]
    [SerializeField] private bool useGravityFusion = true;
    [SerializeField] private SensorManager sensorManager;


    private bool _hasCoarsePose = false;
    private Vector3 _coarsePosition;
    private Quaternion _coarseRotation;
    private Vector3? _coarseHostGravity;


    // We'll reuse this buffer from the last attempt
    private List<Vector3> _preRotatedHostPoints = new List<Vector3>();


    // --- Unity Methods ---
    void Start()
    {
        UnityEngine.Debug.Log("PacketReceiver: starting thread");
        // --- REFACTOR: Bind a raw Socket for zero-alloc receives
        _receiveSocket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        _receiveSocket.Bind(new IPEndPoint(IPAddress.Any, SocketConfig.Port));

        running = true;
        receiveThread = new Thread(ReceiveLoop);
        receiveThread.IsBackground = true;
        receiveThread.Start();
        receiverCamera = Camera.main;
        localMaxDistanceSqr = localMaxDistance * localMaxDistance;
    }

    void OnDestroy()
    {
        running = false;
        _receiveSocket?.Close();
        receiveThread?.Join();
    }

    // --- REFACTOR: Buffer Pool Methods ---
    private byte[] RentBuffer()
    {
        if (bufferPool.TryTake(out byte[] buffer))
        {
            return buffer;
        }
        return new byte[1400]; // MTU size
    }

    private void ReturnBuffer(byte[] buffer)
    {
        bufferPool.Add(buffer);
    }

    private void ReceiveLoop()
    {
        EndPoint remoteEP = new IPEndPoint(IPAddress.Any, 0);
        while (running)
        {
            byte[] buffer = RentBuffer(); // Get a buffer from the pool
            try
            {
                // Receive *into* our existing buffer
                int bytesReceived = _receiveSocket.ReceiveFrom(buffer, ref remoteEP);

                // Read the header to check the type
                PacketUtils.ReadHeader(buffer, 0, out int deviceId, out _, out _, out PacketType type);

                if (type == PacketType.Stop_ICP_Stream)
                {
                    // Handle this simple packet immediately on this thread
                    if (arCloudAnchorManager != null)
                    {
                        arCloudAnchorManager.StopStreamingToPeer(deviceId);
                    }
                    ReturnBuffer(buffer); // Return buffer to pool
                }
                else
                {
                    // Enqueue the work item for the main thread
                    queue.Enqueue(new ReceiveWorkItem
                    {
                        Buffer = buffer,
                        Length = bytesReceived,
                        ReceiveTimestamp = SocketConfig.TimeNowMs() // Use config helper
                    });
                }
            }
            catch (SocketException ex)
            {
                if (running) { UnityEngine.Debug.LogWarning($"SocketException in ReceiveLoop: {ex.Message}"); }
                ReturnBuffer(buffer); // Always return the buffer on error
            }
        }
    }


    void Update()
    {
        while (queue.TryDequeue(out var workItem))
        {
            // Read the header
            int offset = PacketUtils.ReadHeader(workItem.Buffer, 0,
                out int deviceId, out int sequence,
                out double sendTime, out PacketType type);

            // Set the receive time (we can add this to ReadHeader later if needed)
            double receiveTime = workItem.ReceiveTimestamp;

            // Check if we should discard (this logic is good)
            if (arCloudAnchorManager.isHardAnchorResolved &&
               (type == PacketType.ICP_CoarsePose || type == PacketType.PointCloud))
            {
                ReturnBuffer(workItem.Buffer); // Don't forget to return!
                continue;
            }

            StartReceiveProcessingTimer(type, sequence); // Pass type

            // Pass the buffer and data to the handlers
            switch (type)
            {
                case PacketType.CloudAnchorId:
                    HandleCloudAnchorIdPacket(workItem.Buffer, offset, workItem.Length, sequence);
                    break;
                case PacketType.PointCloud:
                    HandlePointCloudPacket(workItem.Buffer, offset, workItem.Length, sequence, sendTime);
                    break;
                case PacketType.ArObjectTransform:
                    HandleARObjectTransformPacket(workItem.Buffer, offset, workItem.Length, sequence);
                    break;
                case PacketType.ICP_CoarsePose:
                    HandleICPAnchorPosePacket(workItem.Buffer, offset, workItem.Length, sequence, sendTime);
                    break;
                default:
                    UnityEngine.Debug.Log($"Receiving Unknown Packet Type {type}");
                    break;
            }

            // CRITICAL: Return the buffer to the pool after processing
            ReturnBuffer(workItem.Buffer);
        }
    }


    private void StartReceiveProcessingTimer(PacketType type, int sequenceId)
    {
        string eventName = type switch
        {
            PacketType.ArObjectTransform => "Receiver_Proc_Asset_Update",
            PacketType.CloudAnchorId => "Receiver_Proc_Anchor_Resolve",
            PacketType.PointCloud => "Receiver_Proc_PointCloud",
            PacketType.ICP_CoarsePose => "Receiver_Proc_ICP_Anchor",
            _ => null
        };
        if (eventName != null)
        {
            LatencyLogger.StartEvent(eventName, sequenceId);
        }
    }



    private void HandleICPAnchorPosePacket(byte[] buffer, int offset, int length, int sequence, double sendTime)
    {
        // --- 1. DECLARE ALL VARIABLES ---
        Vector3 coarsePosition = Vector3.zero;
        Quaternion coarseRotation = Quaternion.identity;
        Vector3 hostGravity = Vector3.zero;
        int payloadLength = length - offset;
        bool hasGravityData = false;

        // --- 2. TRY TO READ THE PACKET ---
        if (useGravityFusion && payloadLength >= SerializationUtils.CoarsePoseGravitySize)
        {
            // Try to read the 40-byte gravity payload
            if (SerializationUtils.ReadPoseWithGravity(buffer, offset, payloadLength,
                out coarsePosition, out coarseRotation, out hostGravity) != -1)
            {
                hasGravityData = true;
            }
            // If it failed (returned -1), 'coarsePosition' is still unassigned.
            // We must fall through to the next 'if' block.
        }


        if (!hasGravityData)
        {
            if (SerializationUtils.ReadTransform(buffer, offset, out coarsePosition, out coarseRotation) == -1)
            {
                UnityEngine.Debug.LogError($"[ICP] Failed to deserialize CoarsePose packet (Seq {sequence}).");
                return;
            }
        }

        // 1. Store the coarse pose data. This is its ONLY job.
        _hasCoarsePose = true;
        _coarsePosition = coarsePosition;
        _coarseRotation = coarseRotation;
        _coarseHostGravity = hasGravityData ? hostGravity : (Vector3?)null;

        if (arCloudAnchorManager != null)
        {
            // 2. ONLY instantiate the anchor if it doesn't exist.
            //    NEVER move it here.
            if (arCloudAnchorManager.ICPAnchor == null)
            {
                arCloudAnchorManager.InstantiateICPAnchor(coarsePosition, coarseRotation);
            }

            if (_pendingPointClouds.TryGetValue(sequence, out var pendingPoints))
            {
                // 3. Process the points (this will now move the anchor)
                ProcessPointCloud(pendingPoints, sequence, sendTime,
                    _coarseHostGravity,
                    _coarsePosition,
                    _coarseRotation
                );

                _pendingPointClouds.Remove(sequence);
            }
        }
    }



    private void HandlePointCloudPacket(byte[] buffer, int offset, int length, int sequence, double sendTime)
    {
        // 1. Deserialize the points into our reusable buffer
        _relativePointsBuffer.Clear();
        SerializationUtils.ReadPointCloud(buffer, offset, length - offset, _relativePointsBuffer);

        if (_hasCoarsePose)
        {
            // Yes, we have it. Process the points immediately
            // using our stored "base" pose.
            ProcessPointCloud(_relativePointsBuffer, sequence, sendTime,
                _coarseHostGravity,
                _coarsePosition,
                _coarseRotation
            );
        }
        else
        {
            // No, the pose hasn't arrived. Buffer these points
            // to be processed when the pose packet *does* arrive.
            _pendingPointClouds.TryAdd(sequence, new List<Vector3>(_relativePointsBuffer));
        }
    }

    private void ProcessPointCloud(List<Vector3> relativePoints_Received, int sequence, double sendTime,
                               Vector3? hostGravity, Vector3 coarsePosition, Quaternion coarseRotation)
    {
        // 1. Get local (receiver's) full point cloud
        _localPointsBuffer.Clear();
        GetLocalFeaturePoints(_localPointsBuffer);

        // 2. Get the coarse pose (blue sphere) from the anchor manager
        // Transform coarseAnchorTransform = arCloudAnchorManager.ICPAnchor.transform;
        // Vector3 coarsePosition = coarseAnchorTransform.position;
        // Quaternion coarseRotation = coarseAnchorTransform.rotation;

        // 3. Filter the local cloud to a small, relevant region (in-view and nearby)
        _filteredLocalPoints.Clear();
        FrustumCull(_localPointsBuffer, _filteredLocalPoints);

        // 4. Create the host (sender's) point cloud in world space using the coarse pose
        _hostPointsWorld.Clear();
        foreach (Vector3 relativePoint in relativePoints_Received)
        {
            _hostPointsWorld.Add((coarseRotation * relativePoint) + coarsePosition);
        }

        // 5. Check if we have enough data in *both* clouds to attempt a match
        if (_hostPointsWorld.Count > 0 && _filteredLocalPoints.Count > 0)
        {
            // 6. Run the "Master Solver"
            // This is the core of our refactor. We pass all available data to the
            // solver, and *it* decides how to use it.
            Vector3? localGravity = (useGravityFusion && sensorManager != null) ?
                                    sensorManager.FilteredGravity :
                                    (Vector3?)null;


            LatencyLogger.StartEvent("GravityFusionCalculation", sequence);
            (Matrix4x4 alignmentTransform, float alignmentError) = RunConfiguredSolver(
                _hostPointsWorld,        // The "coarse" host cloud
                _filteredLocalPoints,    // The filtered local cloud
                hostGravity,             // The host's gravity (or null)
                localGravity             // Our gravity (or null)
            );
            LatencyLogger.EndEvent("GravityFusionCalculation", sequence);


            // --- SAFETY CHECKS ---

            // A. Check for NaN (solver failure)
            if (float.IsNaN(alignmentError) || float.IsNaN(alignmentTransform[0, 0]))
            {
                UnityEngine.Debug.LogError($"[ICP] Solver failed and returned NaN! Discarding result.");
                _lastKnownHostTransform = null; // Force a full RANSAC reset next time
                return; // Drop this packet
            }

            // 8. Apply Final Transform
            // The 'alignmentTransform' is the delta from the original coarse cloud
            // to the local cloud (it already includes any gravity correction).
            Matrix4x4 coarsePoseMatrix = Matrix4x4.TRS(coarsePosition, coarseRotation, Vector3.one);
            Matrix4x4 refinedPoseMatrix = alignmentTransform * coarsePoseMatrix;

            
            // 9. State/Error Check (for stateful tracking)
            if (alignmentError > icpErrorResetThreshold)
            {
                _lastKnownHostTransform = null; // Force RANSAC reset
                UnityEngine.Debug.LogWarning($"[ICP] High error ({alignmentError:F4}m). Forcing RANSAC re-localization.");
            }
            else
            {
                // Good transform, save it as the guess for the next frame
                _lastKnownHostTransform = alignmentTransform;
            }


            // A) Move the Anchor Sphere
            ApplyIcpTransform(refinedPoseMatrix);
            LogSoftColocalizationLatency(sendTime, sequence);

            // B) Re-calculate the point cloud's world position using the *new* refined pose
            // This is critical to keep the points and sphere in sync.
            _hostPointsWorld.Clear();
            Vector3 refinedPosition = refinedPoseMatrix.GetColumn(3);
            Quaternion refinedRotation = refinedPoseMatrix.rotation;
            IcpStabilityLogger.LogSample(sequence, refinedPosition, refinedRotation, alignmentError);

            foreach (Vector3 relativePoint in relativePoints_Received)
            {
                _hostPointsWorld.Add((refinedRotation * relativePoint) + refinedPosition);
            }

            // C) Re-Visualize the point cloud
            if (hostVisualizer_CPU != null)
            {
                stopwatch.Reset();
                stopwatch.Start();
                hostVisualizer_CPU.UpdatePoints(_hostPointsWorld, Color.green);
                stopwatch.Stop();
                UnityEngine.Debug.Log($"ICP - Visualizer time: {stopwatch.ElapsedMilliseconds}");
            }
        }
        else
        {
            // This is our data pre-condition check
            UnityEngine.Debug.LogWarning($"[ICP] Skipping ICP: Source or Target point cloud is empty (Source: {_hostPointsWorld.Count}, Target-Filtered: {_filteredLocalPoints.Count}).");
        }
    }



    // Replace your old FrustumCull with this
    // In PacketReceiver.cs
    private void FrustumCull(List<Vector3> allLocalPoints, List<Vector3> filteredPoints)
    {
        filteredPoints.Clear();
        if (receiverCamera == null)
        {
            UnityEngine.Debug.LogError("[ICP] No camera reference. Frustum culling failed.");
            foreach (var p in allLocalPoints) { filteredPoints.Add(p); } // Return full list
            return;
        }

        // --- REFACTOR ---
        // Get the camera's position *once* outside the loop
        Vector3 cameraPosition = receiverCamera.transform.position;

        foreach (Vector3 localPoint in allLocalPoints)
        {
            // 1. Frustum Check
            Vector3 viewportPoint = receiverCamera.WorldToViewportPoint(localPoint);

            if (viewportPoint.z > 0 &&
                viewportPoint.x > 0 && viewportPoint.x < 1 &&
                viewportPoint.y > 0 && viewportPoint.y < 1)
            {
                // 2. --- NEW: Distance Check ---
                // If it's in the frustum, *also* check if it's close enough.
                if ((localPoint - cameraPosition).sqrMagnitude < localMaxDistanceSqr)
                {
                    filteredPoints.Add(localPoint);
                }
            }
        }

        UnityEngine.Debug.Log($"[ICP Pruning] Pruned local points. Total: {allLocalPoints.Count} -> Frustum+Dist: {filteredPoints.Count} points.");
    }



    private (Matrix4x4, float) RunConfiguredSolver(List<Vector3> hostPoints, List<Vector3> localPoints,
                                                 Vector3? hostGravity, Vector3? localGravity)
    {
        // --- 1. SENSOR FUSION PATH (FAST & ROBUST) ---
        bool canUseGravityFusion = useGravityFusion &&
                                   hostGravity.HasValue &&
                                   localGravity.HasValue;

        if (canUseGravityFusion)
        {
            UnityEngine.Debug.Log("[ICP] Running Gravity-Assisted 4-DoF Hierarchy.");
            // LatencyLogger.StartEvent("GravityFusionCalculation", );

            // LEVEL 1: Gravity Assist (Solve Pitch/Roll)
            Quaternion gravityCorrection = Quaternion.FromToRotation(hostGravity.Value, localGravity.Value);

            _preRotatedHostPoints.Clear();
            foreach (var p in hostPoints)
            {
                _preRotatedHostPoints.Add(gravityCorrection * p);
            }
            // We use _preRotatedHostPoints and localPoints from here on.

            // --- THIS IS THE NEW LOGIC ---
            (Matrix4x4 solverTransform_4DoF, float solverError) = (Matrix4x4.identity, 0f);

            // if (_lastKnownHostTransform.HasValue)
            // {
            //     // LEVEL 2 (STATEFUL): We have a pose from last frame. Just refine it.
            //     // This is fast and stable, preventing jumps.
            //     (solverTransform_4DoF, solverError) = SimpleICP.Solve_4DoF(
            //         _preRotatedHostPoints,
            //         localPoints,
            //         maxIterations: 10,
            //         initialTransform: _lastKnownHostTransform.Value // <-- Use state
            //     );
            // }
            // else
            // {
            // LEVEL 2 (STATELESS): We are lost. Run RANSAC once to find a new pose.

            // A. Get robust guess with 4-DoF RANSAC
            stopwatch.Reset();
            stopwatch.Start();
            (Matrix4x4 ransacGuess, float ransacError) = SimpleICP.Solve_RANSAC_4DoF_stick(
                _preRotatedHostPoints,
                localPoints,
                ransacIterations: 6000,
                // inlierThreshold: 0.01f // 
                inlierThreshold: 0.04f // 
                                       // inlierThreshold: 0.005f
            );
            stopwatch.Stop();
            UnityEngine.Debug.Log($"ICP RANSAC Solve Time: {stopwatch.ElapsedMilliseconds}");

            // B. Refine that guess
            stopwatch.Reset();
            stopwatch.Start();
            // (solverTransform_4DoF, solverError) = SimpleICP.Solve_4DoF_Anneal(
            //     _preRotatedHostPoints,
            //     localPoints,
            //     maxIterations: 15,
            //     initialTransform: ransacGuess
            // );
            (solverTransform_4DoF, solverError) = SimpleICP.Solve_4DoF(
                _preRotatedHostPoints,
                localPoints,
                maxIterations: 15,
                initialTransform: ransacGuess
            );
            stopwatch.Stop();
            UnityEngine.Debug.Log($"ICP 4DOF Solve Time: {stopwatch.ElapsedMilliseconds}");
            // }

            // LEVEL 4: Combine Transforms
            Matrix4x4 gravityTransform = Matrix4x4.TRS(Vector3.zero, gravityCorrection, Vector3.one);
            Matrix4x4 finalAlignment = solverTransform_4DoF * gravityTransform;

            return (finalAlignment, solverError);
        }

        // --- 2. NO-SENSOR FALLBACK PATH (6-DoF) ---
        // (This path is unchanged and is already stateful)

        UnityEngine.Debug.Log("[ICP] No sensors. Running full 6-DoF solver.");
        if (!useRansacSolver)
        {
            return SimpleICP.Solve_Simple(hostPoints, localPoints);
        }
        else if (_lastKnownHostTransform.HasValue)
        {
            return SimpleICP.Solve_Simple(
                hostPoints, localPoints,
                maxIterations: 10, initialTransform: _lastKnownHostTransform.Value
            );
        }
        else
        {
            return SimpleICP.Solve_Robust(
                hostPoints, localPoints,
                ransacIterations: 5000, inlierThreshold: 0.04f, simpleIterations: 10
            );
        }
    }


    private void HandleCloudAnchorIdPacket(byte[] buffer, int offset, int length, int sequence)
    {
        if (arCloudAnchorManager == null)
        {
            UnityEngine.Debug.LogError("arCloudAnchorManager is null. Cannot resolve ID.");
            return;
        }
        // --- REFACTOR: Use new utility
        int payloadLength = length - offset;
        if (SerializationUtils.ReadString(buffer, offset, payloadLength, out string anchorId))
        {
            arCloudAnchorManager.StartResolveProcess(anchorId);
        }
        else
        {
            UnityEngine.Debug.LogError("Failed to deserialize CloudAnchorId.");
        }
        LatencyLogger.EndEvent("Receiver_Proc_Anchor_Resolve", sequence);
    }

    private void ApplyIcpTransform(Matrix4x4 newWorldTransform)
    {
        if (arCloudAnchorManager == null || arCloudAnchorManager.ICPAnchor == null) { return; }
        Transform anchorTransform = arCloudAnchorManager.ICPAnchor.transform;
        anchorTransform.position = newWorldTransform.GetColumn(3);
        anchorTransform.rotation = newWorldTransform.rotation;
        anchorTransform.name = "ICP_Anchor (Refined)";
    }


    private void GetLocalFeaturePoints(List<Vector3> outPoints)
    {
        if (localPointCloudProvider == null)
        {
            UnityEngine.Debug.LogError("Local PointCloudProvider is missing!");
            return;
        }
        // This new method will fill 'outPoints' without creating a new list
        localPointCloudProvider.GetLatestWorldPoints(outPoints);
    }

    private void LogSoftColocalizationLatency(double sendTimestamp, int sequence)
    {
        double T_A_send_ms = sendTimestamp;
        double T_B_applied_ms = SocketConfig.TimeNowMs(); // --- REFACTOR: Use config helper
        double durationMs = T_B_applied_ms - T_A_send_ms;
        long durationTicks = (long)(durationMs * System.Diagnostics.Stopwatch.Frequency / 1000.0);
        LatencyLogger.LogExternalDuration("Soft_Colocalization_E2E", sequence, durationTicks);
    }

    // private void HandleARObjectTransformPacket(byte[] buffer, int offset, int length, int sequence)
    // {
    //     // 1. --- (UNCHANGED) ---
    //     // This business logic (checking prefabs and instantiating
    //     // the object if it's the first time) remains exactly the same.
    //     if (arCloudAnchorManager == null || arCloudAnchorManager.ReceivedAnchorPrefab == null)
    //     {
    //         LatencyLogger.EndEvent("Receiver_Proc_Asset_Update", sequence);
    //         return;
    //     }
    //     if (activeReplicatedObject == null)
    //     {
    //         if (replicatedArObjectPrefab == null)
    //         {
    //             LatencyLogger.EndEvent("Receiver_Proc_Asset_Update", sequence);
    //             return;
    //         }
    //         activeReplicatedObject = Instantiate(replicatedArObjectPrefab, arCloudAnchorManager.ReceivedAnchorPrefab.transform);
    //     }

    //     // 2. --- (THIS IS THE KEY CHANGE) ---
    //     // We now use our new, unified serialization utility.
    //     // It reads *directly* from the main 'buffer' at the
    //     // 'offset' (which is 20, right after the header).
    //     if (SerializationUtils.ReadTransform(buffer, offset, out Vector3 relativePosition, out Quaternion relativeRotation) == -1)
    //     {
    //         // Our new ReadTransform returns -1 if the payload is too small,
    //         // which is cleaner and faster than a try/catch block.
    //         UnityEngine.Debug.LogError("Failed to deserialize AR payload: Payload too small.");
    //         LatencyLogger.EndEvent("Receiver_Proc_Asset_Update", sequence);
    //         return;
    //     }

    //     // 3. --- (UNCHANGED) ---
    //     // This logic is the same as before. It just uses the
    //     // 'relativePosition' and 'relativeRotation' we got
    //     // from the new ReadTransform function.
    //     Transform anchorTransform = arCloudAnchorManager.ReceivedAnchorPrefab.transform;
    //     Vector3 worldPosition = anchorTransform.TransformPoint(relativePosition);
    //     Quaternion worldRotation = anchorTransform.rotation * relativeRotation;
    //     activeReplicatedObject.transform.position = worldPosition;
    //     activeReplicatedObject.transform.rotation = worldRotation;
    //     if (!activeReplicatedObject.activeSelf)
    //     {
    //         activeReplicatedObject.SetActive(true);
    //     }

    //     // 4. --- (MINOR CHANGE) ---
    //     // We use the 'sequence' argument passed into the function,
    //     // instead of 'packet.Sequence'.
    //     LatencyLogger.EndEvent("Receiver_Proc_Asset_Update", sequence);
    // }
    private void HandleARObjectTransformPacket(byte[] buffer, int offset, int length, int sequence)
    {
        Transform parentAnchor = arCloudAnchorManager.CurrentWorldAnchor;

        // If we haven't resolved the anchor yet, we can't place the object relative to it.
        if (arCloudAnchorManager == null || parentAnchor == null)
        {
            LatencyLogger.EndEvent("Receiver_Proc_Asset_Update", sequence);
            return;
        }

        if (activeReplicatedObject == null)
        {
            if (replicatedArObjectPrefab == null) return;
            
            // Instantiate as a CHILD of the live Anchor
            activeReplicatedObject = Instantiate(replicatedArObjectPrefab, parentAnchor);
        }
        else
        {
            // Safety: Ensure it stays attached to the anchor (e.g., if we switched from ICP -> Cloud)
            if (activeReplicatedObject.transform.parent != parentAnchor)
            {
                activeReplicatedObject.transform.SetParent(parentAnchor, false);
            }
        }

        if (SerializationUtils.ReadTransform(buffer, offset, out Vector3 relativePosition, out Quaternion relativeRotation) == -1)
        {
            return;
        }

        // Apply to LOCAL Transform
        // Since we are a child of the anchor, LocalPosition is exactly what we want.
        activeReplicatedObject.transform.localPosition = relativePosition;
        activeReplicatedObject.transform.localRotation = relativeRotation;

        if (!activeReplicatedObject.activeSelf) activeReplicatedObject.SetActive(true);

        LatencyLogger.EndEvent("Receiver_Proc_Asset_Update", sequence);
    }
}